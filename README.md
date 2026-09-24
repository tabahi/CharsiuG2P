# Standard G2P

Multilingual text → **standardized, indexed phoneme targets** for training speech models.

This is a fork of [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) (Zhu, Zhang & Jurgens, 2022). The upstream
ByT5 model converts a word in any of 100 languages into an IPA string. This fork adds `standard_g2p/`, which turns
those IPA strings into a fixed label space a speech model can be trained on. The label space is the same for every
language, and every index comes with the size of the space it belongs to.

```
text ──► words ──► CharsiuG2P ──► raw IPA ──► normalize ──► segment ──► layers ──► gold inventory ──► group inventory
       split / word  (ByT5)       per word    (artifacts)   (phonemes)  tone/stress   270 tokens        per-group softmax
       segmentation
                                                                         /length
```

---

## Quick start

```python
from standard_g2p.gold_g2p import goldG2P
from standard_g2p import group_inventory as GI

G = goldG2P(device='cuda:0')        # weights are fetched once into tmp/
d = G.phonemize_sentence('uma parceria', lang='pt-BR')
```

```python
{'lang': 'pt-BR', 'g2p_lang': 'por-bz',
 'n_tones': 22, 'n_stresses': 3, 'n_lengths': 3, 'n_gold_ph': 270, 'n_gold_phg': 15,   # scope
 'words':    ['uma', 'parceria'],
 'ipa':      ['ũɐ', 'paʁseɾiɐ'],                                     # raw model output, for audit
 'phonemes': ['ũ', 'ɐ', 'p', 'a', 'ʁ', 's', 'e', 'ɾ', 'i', 'ɐ'],    # standardized IPA segments
 'tone':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                         # < n_tones
 'stress':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                         # < n_stresses
 'length':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                         # < n_lengths
 'word_num': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 'gold_ph':  [204, 38, 23, 4, 37, 5, 7, 18, 6, 38],                  # < n_gold_ph
 'gold_phg': [1, 2, 4, 3, 8, 7, 2, 11, 1, 2],                        # < n_gold_phg
 'gold_unmapped': []}
```

To get the indices for the model of that language's group:

```python
GI.to_local(d['gold_ph'], 'pt-BR')
```

```python
{'lang': 'pt-BR', 'group': 'latin',
 'local_ph': [177, 38, 23, 4, 37, 5, 7, 18, 6, 38],   # < n_local_ph
 'n_local_ph': 213, 'n_gold_ph': 270,
 'unk': 3, 'n_folded': 0, 'gold_fingerprint': '9438371ed6dd'}
```

`GI.decode(t['local_ph'], 'pt-BR')` turns local indices back into phonemes.

### Examples

| script | model? | shows |
|---|---|---|
| [examples/01_phonemize_sentence.py](examples/01_phonemize_sentence.py) | yes | sentence → layers → group targets, across 7 languages |
| [examples/02_ipa_layers.py](examples/02_ipa_layers.py) | no | how raw IPA is normalized, segmented and split into layers |
| [examples/03_inventories.py](examples/03_inventories.py) | no | head sizes, the gold inventory, the per-group tables |
| [examples/04_features.py](examples/04_features.py) | no | articulatory features for an auxiliary head |
| [examples/05_phonemize_srt.py](examples/05_phonemize_srt.py) | yes | transcript JSON in, `.gs.json` out |
| [examples/06_word_segmentation.py](examples/06_word_segmentation.py) | yes | th/km/my/ja/zh/yue: text → words → IPA with tone |

Requirements: `torch`, `transformers`, `huggingface_hub` for inference, plus a segmenter for each unspaced
language you use (see the table under *Know before you trust the output*). The tables
(`lang_codes`, `group_inventory`, `phoneme_inventory_gold`, `phoneme_features`) and `word_segmentation.split_words`
need only the standard library, so a training data loader can import them without pulling in `transformers`.

---

## How the standardization works

The model returns one IPA **string** per word. Four problems stand between that string and a training label. Each
one corrupts the labels silently if it is left alone.

### 1. Normalization: the training data is not uniformly IPA

CharsiuG2P was trained on dictionaries scraped largely from Wiktionary. About 5% of the tokens are transcription
artifacts, and the model reproduces them at inference. `normalize_ipa()` repairs them before anything is counted:

| artifact | where | example | becomes |
|---|---|---|---|
| SAMPA instead of IPA | `swe` (all of it) | `plA:na%vE:gen` | `plɑːnaˌvɛːɡen` |
| SAMPA capitals | `uzb`, `ger`, `swa` | `t͡S` | `t͡ʃ` |
| Chao tone digits | `nan` | `kʰuan²¹⁻⁵³` | tone layer `21` |
| optional-palatalization parens | `rus` | `⁽ʲ⁾` | `ʲ` |
| codepoint duplicates | everywhere | `:` `g` `ʧ` | `ː` `ɡ` `t͡ʃ` |
| Greek look-alikes | `fra-qu`, `grc` | `ε` | `ɛ` |
| above/below diacritic variants | | `ŋ̊` / `n̥` | one spelling |
| doubled modifiers | `ara` | `tˤˤ` | `tˤ` |

### 2. Segmentation: an IPA string is not a list of phonemes

One phoneme can span several codepoints: `t͡ʃ` (tie bar), `pʰ` (modifier letter), `ẽ` (combining mark), `aː` (length
mark). `segment_ipa()` groups codepoints by Unicode category, so each of these stays one segment.

### 3. Layers: tone, stress and length are not phonemes

These belong to the syllable, not the segment. Folding them into the phoneme label multiplies the inventory (`a`,
`aː`, `a˧`, `aː˥˩` would each be a class) and puts tonal languages in a label space of their own. `decompose_ipa()`
splits them into parallel arrays of the same length:

```python
>>> decompose_ipa('pʰaː˧.saː˩˩˦', lang='tha')
{'segments': ['pʰ', 'a', 's', 'a'],
 'length':   [ 0,    2,   0,   2 ],      # 0 short / 1 half-long / 2 long
 'tone':     ['',   '˧',  '', '˩˩˦'],    # on the nucleus; -> TONE_VOCAB index
 'stress':   [ 0,    0,   0,   0 ]}      # 0 none / 1 primary / 2 secondary
```

Tone contours from all scripts are unified into Chao numerals (`˧˥` and `³⁵` both become `35`) and indexed into
`TONE_VOCAB` (22 values, 0 = no tone). This way Mandarin, Thai and Cantonese share their segment classes with English
and Arabic. Cantonese alone has 31 segments and 23 tone contours: merged, that would be hundreds of classes; as
layers it is 31 + 23.

### 4. The gold inventory: one closed label set for all languages

[standard_g2p/phoneme_inventory_gold.py](standard_g2p/phoneme_inventory_gold.py) is built from the whole CharsiuG2P
training corpus (7.7M pronunciations, 100 languages). A segment is admitted if it is attested in **at least 2
languages**. A phoneme seen in only one language has no cross-lingual evidence, so the model would just memorize
that language's data.

| admission criterion | segments | token coverage |
|---|---:|---:|
| everything | 676 | 100% |
| **≥ 2 languages** | **266** | **99.47%** |
| ≥ 3 languages | 207 | 98.81% |
| ≥ 5 languages | 154 | 98.20% |

Token layout: `<blank>` (0, for CTC), `SIL`, `noise`, `<unk>`, then the 266 phonemes by frequency, **270 tokens** in
total. A segment outside the inventory is rewritten by `BACKOFF` (391 entries). In order, it tries to: peel off fine
diacritics → reduce to the bare base → split a tie-bar unit → apply a manual table (implosives `ɓ → b`) → route junk
to `noise`. With backoff, coverage is 100%. Anything that still has no mapping becomes `<unk>` and is listed in
`gold_unmapped`.

`gold_ph` is **not** index-aligned with `phonemes`, because a backed-off segment can expand into several units
(`ʈ͡ʂ → ʈ ʂ`).

**Broad groups and features.** `gold_phg` gives each gold token one of 15 broad classes (vowel_close, stop_voiced,
nasal, …), for a coarse head trained alongside the fine one.
[standard_g2p/phoneme_features.py](standard_g2p/phoneme_features.py) gives every token 20 articulatory features
(taken from [panphon](https://github.com/dmort27/panphon) and stored in the repo, so panphon is not needed at run
time). `feature_targets()` returns binary targets plus a mask. The mask covers features that don't apply (`distr` on a
vowel) and special tokens, so no loss is taken there.

### 5. Language groups: a smaller softmax per model

Models are multilingual *within* a processing group. Groups follow the **preprocessing path**: writing system, word
segmentation and tone. They do not follow language family. Grouping by family was measured and rejected, because
phoneme inventories do not recover families (see
[standard_g2p/mappings/lang_stats.md](standard_g2p/mappings/lang_stats.md) §4).

Each group predicts a **subset** of the 270 gold tokens. For every member language, and for each of two sources (the
model's actual FLEURS output and the training dictionaries), the most frequent gold phonemes are kept until 99.9% of
that language's tokens are covered. The group list is the union of these. Every language therefore keeps ≥ 99.9% of
its tokens, and a phoneme that one small language needs is not voted out by a large one.

| group | tokens | | group | tokens |
|---|---:|---|---|---:|
| `latin` | 213 | | `brahmic` | 98 |
| `cyrillic` | 128 | | `cjk` | 48 |
| `other_alphabetic` | 107 | | `thai_khmer` | 47 |
| `abjad` | 91 | | `japanese` | 28 |

The special tokens keep their gold indices in every group (blank = 0 everywhere). A gold token outside a group's
list becomes that group's `<unk>`, and `to_local()` reports how many did so as `n_folded`. Stored files keep only gold
indices and are converted at load time. That way the group lists can be regenerated without rewriting a corpus.
`GI.load()` raises `StaleInventoryError` if the group table was built against a different gold inventory.

### Scope: every index carries its range

| key | indexes | size |
|---|---|---|
| `tone` | `gold_g2p.TONE_VOCAB` | `n_tones` = 22 |
| `stress` | none / primary / secondary | `n_stresses` = 3 |
| `length` | short / half-long / long | `n_lengths` = 3 |
| `gold_ph` | `phoneme_inventory_gold.TOKENS` | `n_gold_ph` = 270 |
| `gold_phg` | `phoneme_features.GROUP_NAMES` | `n_gold_phg` = 15 |
| `local_ph` (from `to_local`) | `group_inventory.tokens(group)` | `n_local_ph` (per group) |

The same scope keys appear in `phonemize_words` / `phonemize_sentence` output and in the header of every `.gs.json`
written by `phonemize_srt`.

---

## Language codes

`lang` is always a single **BCP 47** code: ISO 639-1 where the language has one (`en`, `zh`), otherwise ISO 639-3
(`ckb`, `hbs`). A suffix selects a non-default variant (`pt-BR`, `es-419`, `zh-Hant`, `en-GB`). CharsiuG2P's own tags
(`eng-us`, `ger`, `por-bz`) are internal and never valid as `lang`.

```python
from standard_g2p.lang_codes import bcp47_to_tag, tag_to_bcp47, processing_group
bcp47_to_tag('pt-BR')     # 'por-bz'
bcp47_to_tag('pt')        # 'por-po'  -- a bare code picks the default variant
tag_to_bcp47('eng-uk')    # 'en-GB'
processing_group('cmn')   # 'cjk'
```

A bare code resolves to its default variant: `pt` → European Portuguese, where *parceria* is `pɐɾsɨɾiɐ` rather than
`paʁseɾiɐ`. Nothing errors, so pass the regional code whenever it matters. `bcp47_to_tag` **raises** on an unknown
language instead of falling back. ByT5 reads bytes, so an invented tag still produces IPA-shaped output for some
other language, and nothing downstream could tell.

---

## Know before you trust the output

- **G2P gives dictionary pronunciations, not what was said.** Reduction, coarticulation and dialect make real speech
  differ systematically from these labels.
- **The model reproduces its training dictionary, including casual variants.** For example, `dicts/por-bz.tsv` lists
  both `umɐ` and `ũɐ` for *uma*, and the model returns the casual form `ũɐ` with the /m/ dropped. The model also
  under-predicts rare phonemes, because it drifts toward frequent symbols.
- **Eight languages need word segmentation**: `zh`, `yue`, `nan`, `ja`, `th`, `tts`, `km`, `my`. This is a *word*
  model, and in these scripts `split_words` hands it whole clauses. `word_segmentation.segment(text, lang)` splits
  them into words where a segmenter exists, and `phonemize_sentence` / `phonemize_srt` use it. Space-delimited
  languages go through `split_words` exactly as before.

  | lang | segmenter | install |
  |---|---|---|
  | `th` | pythainlp `newmm`, plus repairs for cuts inside a syllable and for `ๆ` | `pythainlp` |
  | `km` | khmer-nltk, plus `ៗ` expansion | `khmer-nltk` |
  | `my` | pyidaungsu | `pyidaungsu` |
  | `ja` | fugashi with UniDic-lite. **The model gets UniDic's katakana reading, not the text**, so `words` is katakana | `fugashi unidic-lite` |
  | `zh`, `zh-Hant` | longest match on the model's own `dicts/zho-{s,t}.tsv` | none |
  | `yue` | pycantonese | `pycantonese` |

  Each backend was chosen by running the model on real sentences, not by counting how many segmented words are
  dictionary entries. That count favours cutting words into dictionary pieces, and for Thai, Khmer and Burmese those
  pieces are read differently: loanwords turn into letter names, linking vowels are dropped, and Burmese loses the
  consonant voicing across word boundaries. The measurements are in
  [word_segmentation.py](standard_g2p/word_segmentation.py). Some highlights:
  - **Thai** (439 FLEURS dev clips): the old split gave 4.1 "words" per clip, averaging 29 characters. At
    `max_length=64` the model's IPA stopped partway through each one, so the rest of the clause got no labels at
    all. Now there are 24.2 words per clip, and 93.2% of vowels carry a tone.
  - **Japanese:** the reading fixes the particles `は` → `wa` and `へ` → `e` (from the text the model says `ha`,
    `he`), and it fixes readings that depend on context (`昨日` → `kinoː`, `雨` → `ame`). UniDic-lite does read `日本`
    as `nipːoɴ`.
  - **Mandarin:** per-character input reads `音乐` with the `快乐` reading and loses the neutral tone of `们`. The
    longest match does not have either problem.
  - In zh, yue and th every word carries a tone. The vowels without one are the first half of a diphthong; the
    tone goes on the second half.
  - **Not segmented**: `nan` and `tts` (both in `EXCLUDED_ISO`). `has_segmenter(lang)` is False for them, and
    `g2p_task.task_g2p_phonemize` skips them by default.
  - `phonemize_srt` does not use Whisper's `words` directly for these scripts. Whisper's words there are tokenizer
    pieces, so it joins them and segments the joined text. `words` in the output is the segmented list, and
    `word_num` indexes it.
- **Numerals are not verbalized.** `1979` goes to the model as one "word" in every language, and in Thai it comes back
  with no tone.
- **Excluded languages** (`lang_codes.EXCLUDED_ISO`): `my` (its tone is written as vowel diacritics, so it never
  reaches the tone layer), `nan` (no segmenter, sandhi), and `tts` (its dictionary is a romanization, not IPA). They
  keep their group but did not shape its token list. `my` is the only member of `burmese`, so that group's list holds
  only the 4 special tokens and `to_local` turns every Burmese phoneme into `<unk>`. Burmese is segmented now, but
  its labels are unusable until it is taken off the list and `create_phoneme_inventories.py` is rerun.
- **Stress is transcribed in only 36 of the 100 languages.** A missing stress mark means "not annotated", not
  "unstressed", so mask the stress loss for the other languages.
- **Pitch accent is not tone.** `hbs`, `slv`, `san` and `grc` use the same acute and grave marks for pitch accent, and
  `kur` uses them for stress. `tone_diacritics=True` would misread all of these as tone.
- **Homographs** (English *read*, *lead*) get a single pronunciation, because the model sees no context.

### Upgrading older `.gs.json` files

- Header keys were renamed to match the in-memory output: `n_tokens` → `n_gold_ph`, `n_groups` → `n_gold_phg`,
  `n_stress` → `n_stresses`.
- `GI.to_local()` now returns a dict. The indices are under `['local_ph']`.
- Labels for Brahmic, Thai, Khmer and Burmese scripts written before 2026-09-23 are corrupt and must be regenerated.
  `split_words` used to treat vowel signs and viramas as word separators.
- Labels for th, km, my, ja, zh and yue written before 2026-09-24 were phonemized clause by clause (see above) and
  must be regenerated. Burmese `၏ ၍ ၌ ၎` used to be dropped as punctuation.

---

## Repository layout

```
standard_g2p/                   the module
  gold_g2p.py                   goldG2P model wrapper, IPA normalization/segmentation, layers, file I/O
  lang_codes.py                 BCP 47 <-> CharsiuG2P tag, tone/segmentation flags, groups (no deps)
  word_segmentation.py          text -> words: split_words, and segmenters for unspaced scripts
  phoneme_inventory_gold.py     the 270-token gold inventory + BACKOFF          (generated)
  phoneme_features.py           20 articulatory features + 15 broad groups      (generated)
  group_inventory.py            gold indices -> a group model's indices (no deps)
  mappings/
    group_inventories.json      per-group token lists, loaded at run time       (generated)
    phoneme_counts.md           mapping health, backoff/unmapped report, group lists
    lang_stats.md               the measurements behind lang_codes.py's groupings
    *.json                      the raw counts behind the two reports
scripts/                        generators for everything marked (generated)
examples/                       runnable examples
g2p_task.py                     batch-phonemize a paths_list of transcripts to .gs.json

dicts/  data/  sources/  notebooks/  multilingual_results/      inherited from CharsiuG2P
charsiug2p_original_src/        CharsiuG2P's original training/evaluation code
```

### Regenerating the tables

Everything marked *(generated)* comes from a script. Hand edits are lost at the next regeneration.

```bash
python scripts/layers.py                   # 1. enumerate every segment in data/train/  -> tmp/inventory_build/
python scripts/build_inventory.py          # 2. -> standard_g2p/phoneme_inventory_gold.py
python scripts/build_features.py           # 3. -> standard_g2p/phoneme_features.py (needs panphon)
python scripts/measure_lang_groups.py      # -> mappings/lang_stats.{json,md} (needs scipy)
python scripts/create_phoneme_inventories.py [--paths <fleurs paths_list> --metadata-dir <dir>]
                                           # -> mappings/group_inventories.json + counts + phoneme_counts.md
```

Rebuilding the gold inventory changes its fingerprint. Rerun `create_phoneme_inventories.py` afterwards, or
`group_inventory` will refuse the stale table. The shipped group table was built from both sources. Without `--paths`
the script counts only `dicts/`, which gives slightly different lists.

---

## Attribution

This repository is a fork of **[CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)** by Jian Zhu, Cong Zhang and
David Jurgens. The G2P model (`charsiu/g2p_multilingual_byT5_small_100`), the pronunciation dictionaries (`dicts/`),
the train/dev/test splits (`data/`), the source collection (`sources/`), and the original training code
(`charsiug2p_original_src/`, `notebooks/`) are theirs, distributed under the MIT license (see [LICENSE](LICENSE)).
`standard_g2p/`, `scripts/` and `examples/` are the additions of this fork.

If you use this work, please cite the original paper:

```bibtex
@article{zhu2022charsiu-g2p,
  title={ByT5 model for massively multilingual grapheme-to-phoneme conversion},
  author={Zhu, Jian and Zhang, Cong and Jurgens, David},
  url={https://arxiv.org/abs/2204.03067},
  doi={10.48550/ARXIV.2204.03067},
  year={2022}
}
```

The dictionaries were collected by the CharsiuG2P authors from the sources below. **Please also cite the original
sources of any data you use.** Source and license details for each file are in [sources/info](sources/info).

- WikiPron: Lee et al., *Massively Multilingual Pronunciation Modeling with WikiPron*, LREC 2020
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng), with word lists from the
  [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download) (Goldhahn et al., LREC 2012)
- [ipa-dict](https://github.com/open-dict-data/ipa-dict)
- Kurdish: Veisi et al., 2020 (AsoSoft); Ahmadi, 2019
- [Britfone](https://github.com/JoseLlarena/Britfone) (British English)
- [thai-g2p](https://github.com/wannaphong/thai-g2p) (Thai)
- [Santiago Spanish Lexicon](https://www.openslr.org/34/)
- [Sprakbanken Swedish pronunciation dictionary](https://www.openslr.org/29/)

Articulatory features are derived from [panphon](https://github.com/dmort27/panphon) (Mortensen et al., COLING 2016).

The upstream authors note that the Uzbek (`uzb`) dictionary is known to be incorrect.

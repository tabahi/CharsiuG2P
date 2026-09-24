"""Count phonemes in the gold inventory's index space, per language and per
processing group, and derive each group's deterministic token list from them.

Everything is counted in the SAME space the training targets live in: the 270
tokens of phoneme_inventory_gold, reached through the same `decompose_ipa` ->
`map_phoneme` path goldG2P._gold_map uses. A raw segment the inventory cannot
represent is not dropped -- it is counted under how it was handled:

    direct    the segment is itself an inventory token
    backoff   BACKOFF rewrote it onto one or more inventory tokens
    noise     BACKOFF routed it to `noise` (junk: digits, stray script letters)
    unmapped  no entry at all -- it becomes <unk>, and is what a .gs.json
              lists under `gold_unmapped`

A high backoff or unmapped rate is the signal that the gold inventory itself
needs revisiting; phoneme_counts.md ranks the segments behind it.

Two sources, kept separate all the way through because they measure different
things:

    fleurs  the .gs.json files g2p_task.py wrote for the FLEURS dev clips:
            the model's actual output, one vote per phoneme token.
            Read back as written (`gold_ph`), not recomputed, so this is
            exactly what training will see. It is missing the languages
            g2p_task skips for want of word segmentation (zh, yue, ja, th, km).

    dict    CharsiuG2P's own training dictionaries, one vote per word type, for
            every tag a BCP 47 code reaches. Covers the skipped languages and
            the ~40 FLEURS lacks, so no group list is blind to them.

Languages are keyed by BCP 47 code throughout (lang_codes.tag_to_bcp47). The
CharsiuG2P tag appears only as a `g2p_tag` field, the same split the .gs.json
files make between `lang` and `g2p_lang`.

The FLEURS side needs data that lives outside this repo: the paths_list the
.gs.json files were written from, and the directory its '$/...' paths are
relative to. Without --paths only the dictionaries are counted. The shipped
group_inventories.json was built from both sources.

Usage:
    python scripts/create_phoneme_inventories.py                  # dicts only
    python scripts/create_phoneme_inventories.py \
        --paths paths_list_fleurs_dev_1h.json --metadata-dir /path/to/common_dir

To also flag FLEURS locales phonemized with the wrong regional variant, put
the directory holding the FLEURS loader (`fluers.py`, which defines
`fleurs_to_bcp47`) on PYTHONPATH; without it that check is skipped.

Writes to standard_g2p/mappings/:
    lang_counts.json        per-language counts, both sources
    group_counts.json       the same rolled up per processing group
    group_inventories.json  per-group token lists -- the product
    phoneme_counts.md       a readable summary, including the unmapped report
"""
import os
import sys
import json
import argparse
import collections
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2P_DIR = os.path.join(ROOT, 'standard_g2p')
OUT_DIR = os.path.join(G2P_DIR, 'mappings')
sys.path.insert(0, G2P_DIR)

from gold_g2p import decompose_ipa                                  # noqa: E402
import phoneme_inventory_gold as PI                                   # noqa: E402
# The loader checks this fingerprint on read, so both sides must compute it
# the same way -- hence one definition, owned by the loader.
from group_inventory import gold_fingerprint                          # noqa: E402
from lang_codes import (TAG_TO_ISO, ISO_TO_GROUP, PROCESSING_GROUPS,  # noqa: E402
                        EXCLUDED_ISO, tag_to_bcp47, bcp47_to_tag)

CHARSIU_DICTS = os.path.join(ROOT, 'dicts')
GS_EXT = '.gs.json'

OUTCOMES = ('direct', 'backoff', 'noise', 'unmapped')


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

def reachable_tags():
    """BCP 47 code -> CharsiuG2P tag, for every tag some BCP 47 code reaches.

    Two trained tags are unreachable: 'isl' and 'slo' duplicate 'ice' and
    'slk' under the other ISO 639-2 spelling, and bcp47_to_tag('is'/'sk')
    picks the other one. Production can never emit them, so counting their
    dictionaries would double-weight Icelandic and Slovak.
    """
    out = {}
    for tag in sorted(TAG_TO_ISO):
        code = tag_to_bcp47(tag)
        if bcp47_to_tag(code) == tag:
            out[code] = tag
    return out


def lang_aliases(canonical):
    """Every other spelling bcp47_to_tag accepts -> its canonical code.

    Writers do not all emit the canonical form: FLEURS labels en_us 'en-US'
    and cmn_hans_cn 'cmn', and .gs.json files from before the BCP 47 switch
    carry bare ISO 639-3 ('por', 'afr'). A loader normalises through this
    table before looking a language up, rather than failing on a spelling
    lang_codes itself accepts.
    """
    from lang_codes import (BCP47_TO_TAG, BCP47_DEFAULT_ALIASES, ISO_TO_TAG,
                            ISO_639_1, UnsupportedLanguageError)
    cands = (set(BCP47_TO_TAG) | set(BCP47_DEFAULT_ALIASES) | set(ISO_TO_TAG)
             | {ISO_639_1[i] for i in ISO_TO_TAG if i in ISO_639_1})
    out = {}
    for c in sorted(cands):
        try:
            code = tag_to_bcp47(bcp47_to_tag(c))
        except UnsupportedLanguageError:
            continue
        if code != c and code in canonical:
            out[c] = code
    return out


def lang_meta(code, tag):
    iso = TAG_TO_ISO[tag]
    return {'lang': code, 'g2p_tag': tag, 'iso': iso,
            'group': ISO_TO_GROUP[iso], 'excluded': iso in EXCLUDED_ISO}


# ---------------------------------------------------------------------------
# Counting, in gold-inventory space
# ---------------------------------------------------------------------------

def new_counts():
    return {'segments': 0,
            'outcome': collections.Counter(),
            'gold': collections.Counter(),
            'backoff': collections.Counter(),
            'noise': collections.Counter(),
            'unmapped': collections.Counter()}


def classify(seg):
    """(outcome, gold units) for one raw segment. Built on PI.map_phoneme
    itself, not a re-implementation of it, so it cannot drift from what
    goldG2P._gold_map writes."""
    units = PI.map_phoneme(seg)
    if seg in PI.TOKEN_INDEX:
        return 'direct', units
    if units == [PI.UNK]:
        return 'unmapped', units
    if units == [PI.NOISE]:
        return 'noise', units
    return 'backoff', units


def add_segment(c, seg, n=1):
    outcome, units = classify(seg)
    c['segments'] += n
    c['outcome'][outcome] += n
    if outcome != 'direct':
        c[outcome][seg] += n
    return units


def finish(c, **extra):
    """Counters -> plain dicts, most frequent first, for JSON."""
    out = {'segments': c['segments'],
           'outcome': {k: c['outcome'][k] for k in OUTCOMES}}
    for k in ('gold', 'backoff', 'noise', 'unmapped'):
        out[k] = dict(c[k].most_common())
    out.update(extra)
    return out


def count_dict(args):
    """One vote per word type over a CharsiuG2P dictionary. Only the first of
    an entry's comma-separated variants is taken -- what the model was
    trained to emit first. Decomposed exactly as _phonemize_words_tag does."""
    code, tag, dicts_dir = args
    path = os.path.join(dicts_dir, tag + '.tsv')
    if not os.path.exists(path):
        return code, None
    c = new_counts()
    entries = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2 or not parts[1]:
                continue
            entries += 1
            for seg in decompose_ipa(parts[1].split(',')[0], lang=tag)['segments']:
                c['gold'].update(add_segment(c, seg))
    return code, finish(c, word_types=entries)


def gs_path_of(clip, metadata_dir):
    """The .gs.json g2p_task.py wrote for a paths_list entry (same rule)."""
    srt = clip.get('srt') or ''
    if srt.startswith('$/'):
        srt = os.path.join(metadata_dir, srt[2:])
    elif srt and not srt.startswith('/'):
        srt = os.path.join(metadata_dir, srt)
    return srt.replace('.srt.json', GS_EXT).replace('.srt', GS_EXT)


def count_fleurs_locale(args):
    """Every .gs.json of one FLEURS locale.

    Gold counts are read from the file's own `gold_ph`, i.e. what training
    will consume. The raw `phonemes` are re-classified alongside, both to get
    the outcome split and as a consistency check: if re-mapping them does not
    reproduce `gold_ph`, the file was written against a different inventory
    and is reported as stale rather than silently mixed in.
    """
    locale, paths = args
    per_tag = {}
    missing = 0
    for p in paths:
        try:
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
        except OSError:
            missing += 1
            continue
        tag = d.get('g2p_lang') or ''
        r = per_tag.setdefault(tag, {'c': new_counts(), 'files': 0, 'sil': 0,
                                     'stale_segments': 0, 'file_langs': set(),
                                     'n_tokens': set()})
        r['files'] += 1
        r['file_langs'].add(d.get('lang', ''))
        # 'n_tokens' is the pre-rename spelling of 'n_gold_ph'.
        r['n_tokens'].add(d.get('n_gold_ph', d.get('n_tokens')))
        c = r['c']
        for seg in d.get('segments', []):
            expect = []
            for ph in seg.get('phonemes', []):
                if ph == PI.SIL:
                    r['sil'] += 1
                    expect.append(PI.SIL)
                    continue
                expect.extend(add_segment(c, ph))
            got = [PI.TOKENS[i] for i in seg.get('gold_ph', [])]
            if got != expect:
                r['stale_segments'] += 1
            c['gold'].update(t for t in got if t != PI.SIL)
    out = {}
    for tag, r in per_tag.items():
        out[tag] = finish(r['c'], files=r['files'], sil=r['sil'],
                          stale_segments=r['stale_segments'],
                          file_langs=sorted(r['file_langs']),
                          n_tokens=sorted(x for x in r['n_tokens'] if x is not None))
    return locale, out, missing


def gather(dicts_dir=CHARSIU_DICTS, paths_file=None, metadata_dir='',
           workers=32):
    langs = {code: lang_meta(code, tag) for code, tag in reachable_tags().items()}
    for m in langs.values():
        m['dict'] = m['fleurs'] = None
        m['fleurs_locales'] = []

    with Pool(workers) as pool:
        jobs = [(code, m['g2p_tag'], dicts_dir) for code, m in langs.items()]
        for code, rec in pool.imap_unordered(count_dict, jobs):
            langs[code]['dict'] = rec
            if rec is None:
                print(f'  {code:9s} NO DICT', file=sys.stderr)

        fleurs_notes = {'paths_file': paths_file, 'missing_by_locale': {},
                        'mislabelled': []}
        if paths_file:
            fleurs_notes.update(merge_fleurs(pool, langs, paths_file, metadata_dir))
    return langs, fleurs_notes


def merge_fleurs(pool, langs, paths_file, metadata_dir):
    """Run count_fleurs_locale over every locale and file the results under
    the BCP 47 code of the tag the files were ACTUALLY phonemized with.

    That can differ from what the locale should have been. A paths_list built
    before the BCP 47 switch labels pt_br clips 'por', and g2p_task gives the
    clip's lang priority over the srt's own 'por-bz', so Brazilian Portuguese
    comes out as European. Filing by the tag used keeps the counts truthful;
    the mismatch is reported so the files can be regenerated.
    """
    try:
        from fluers import fleurs_to_bcp47
    except ImportError:
        print('  fluers loader not importable: skipping the wrong-variant check',
              file=sys.stderr)
        fleurs_to_bcp47 = {}

    with open(paths_file, encoding='utf-8') as f:
        clips = json.load(f)
    by_locale = collections.defaultdict(list)
    for c in clips:
        p = gs_path_of(c, metadata_dir)
        by_locale[os.path.basename(os.path.dirname(p))].append(p)

    missing = {}
    mislabelled = []
    for locale, per_tag, miss in pool.imap_unordered(count_fleurs_locale,
                                                     sorted(by_locale.items())):
        if miss:
            missing[locale] = miss
        want = fleurs_to_bcp47.get(locale)
        for tag, rec in per_tag.items():
            code = tag_to_bcp47(tag)
            if want and bcp47_to_tag(want) != tag:
                mislabelled.append({'locale': locale, 'expected_lang': want,
                                    'expected_tag': bcp47_to_tag(want),
                                    'g2p_tag': tag, 'files': rec['files']})
            m = langs[code]
            if m['fleurs'] is None:
                m['fleurs'] = rec
            else:        # two locales, one language: never happens in FLEURS
                raise ValueError(f'{locale} and {m["fleurs_locales"]} both '
                                 f'phonemized as {tag}')
            m['fleurs_locales'].append(locale)
    return {'missing_by_locale': dict(sorted(missing.items())),
            'mislabelled': sorted(mislabelled, key=lambda r: r['locale'])}


# ---------------------------------------------------------------------------
# Group rollup
# ---------------------------------------------------------------------------

def rollup(langs, include_excluded=False):
    """Sum per-language counts into per-group tables, per source.

    `n_langs` per token is breadth: how many languages of the group emit it.
    A token frequent in one language only is still kept -- see select() --
    but breadth is reported so that call can be revisited.
    """
    groups = {}
    for g in PROCESSING_GROUPS:
        members = sorted(c for c, m in langs.items() if m['group'] == g)
        used = [c for c in members if include_excluded or not langs[c]['excluded']]
        rec = {'group': g, 'description': PROCESSING_GROUPS[g],
               'langs': used,
               'excluded_langs': [c for c in members if c not in used]}
        for src in ('fleurs', 'dict'):
            tot = new_counts()
            langs_of = collections.defaultdict(set)
            for code in used:
                s = langs[code][src]
                if not s:
                    continue
                tot['segments'] += s['segments']
                tot['outcome'].update(s['outcome'])
                for k in ('gold', 'backoff', 'noise', 'unmapped'):
                    tot[k].update(s[k])
                for k in ('backoff', 'noise', 'unmapped', 'gold'):
                    for seg in s[k]:
                        langs_of[(k, seg)].add(code)
            out = finish(tot, langs=[c for c in used if langs[c][src]])
            out['n_langs'] = {k: {seg: len(langs_of[(k, seg)]) for seg in out[k]}
                              for k in ('gold', 'backoff', 'noise', 'unmapped')}
            rec[src] = out
        groups[g] = rec
    return groups


# ---------------------------------------------------------------------------
# Selection: each group's token list
#
# The group model's softmax is a SUBSET of the 270 gold tokens, not a new
# inventory: the .gs.json files already carry gold indices, and a group only
# needs to know which of them it predicts. So the product is a list of gold
# tokens per group, plus a gold -> local index table.
#
# A token is kept if some member language needs it, judged per language rather
# than on pooled counts. Pooled counts would let one big dictionary decide for
# everyone (the sv dictionary alone is 9M segments, a FLEURS language is ~30k).
# For each language and each source it has, tokens are taken most-frequent
# first until COVERAGE of that language's phoneme tokens is reached; the group
# list is the union. Every member language therefore keeps at least COVERAGE
# of its tokens on either source, and a phoneme important to one small
# language is not voted out by the others. Breadth is reported, not required
# -- the old ">= 2 languages" gate cannot work per group (`japanese` has one).
#
# Deterministic: ties break on gold index, and the list is emitted in gold
# index order, so it depends on the counts and COVERAGE and nothing else.
# ---------------------------------------------------------------------------

COVERAGE = 0.999


def pick(counts, coverage):
    """Most frequent gold phonemes of one language/source until `coverage`
    of its phoneme tokens (specials excluded) is reached."""
    ranked = sorted(((n, t) for t, n in counts.items()
                     if t not in PI.SPECIAL_TOKENS),
                    key=lambda x: (-x[0], PI.TOKEN_INDEX[x[1]]))
    total = sum(n for n, _ in ranked)
    out, acc = [], 0
    for n, t in ranked:
        if acc >= coverage * total:
            break
        out.append(t)
        acc += n
    return out


def covered(counts, keep):
    total = sum(n for t, n in counts.items() if t not in PI.SPECIAL_TOKENS)
    hit = sum(n for t, n in counts.items() if t in keep)
    return round(hit / total, 6) if total else None


def select(langs, groups, coverage=COVERAGE):
    out = {}
    for g, rec in groups.items():
        by = collections.defaultdict(list)
        for code in rec['langs']:
            for src in ('fleurs', 'dict'):
                s = langs[code][src]
                if s:
                    for t in pick(s['gold'], coverage):
                        by[t].append(f'{code}:{src}')
        phonemes = sorted(by, key=PI.TOKEN_INDEX.get)
        tokens = list(PI.SPECIAL_TOKENS) + phonemes
        local = {t: i for i, t in enumerate(tokens)}
        unk = local[PI.UNK]
        keep = set(tokens)
        out[g] = {
            'group': g,
            'description': rec['description'],
            'langs': rec['langs'],
            'excluded_langs': rec['excluded_langs'],
            'n_tokens': len(tokens),
            'tokens': tokens,
            'gold_index': [PI.TOKEN_INDEX[t] for t in tokens],
            # Indexed by gold index. A gold token outside the list folds to
            # the group's <unk>; by construction that is under 1 - coverage
            # of any member language's tokens.
            'gold_to_local': [local.get(t, unk) for t in PI.TOKENS],
            'coverage': {code: {src: covered(langs[code][src]['gold'], keep)
                                for src in ('fleurs', 'dict')
                                if langs[code][src]}
                         for code in rec['langs']},
            # Which language/source put each phoneme on the list, so a
            # surprising entry can be traced to its cause.
            'selected_by': {t: sorted(by[t]) for t in phonemes},
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# A language is flagged in the report above either rate. Backoff is the one
# that matters in practice: BACKOFF was built from the whole CharsiuG2P corpus,
# so `unmapped` only fires on segments the model invents, while backoff
# silently rewrites real ones (Danish d̥ -> d, Korean t͈ -> t).
FLAG_BACKOFF = 0.01
FLAG_UNMAPPED = 0.0001


def pct(n, d, digits=2):
    return f'{100 * n / d:.{digits}f}%' if d else '-'


def md_code(s):
    """Markdown code span that survives a backtick inside the segment --
    SAMPA leaks such as Amharic k` reach the report verbatim."""
    return f'`` {s} ``' if '`' in s else f'`{s}`'


def primary(m):
    """The source that best matches what training sees for a language:
    the model's own FLEURS output where it exists, else the dictionary."""
    return ('fleurs', m['fleurs']) if m['fleurs'] else ('dict', m['dict'])


def write_report(langs, notes, groups, inv, coverage, path):
    L = []
    A = L.append
    A('# Phoneme counts and per-group inventories')
    A('')
    A('Generated by `scripts/create_phoneme_inventories.py` -- do not edit by hand.')
    A('')
    A(f'Everything is counted in the gold inventory\'s index space '
      f'(`phoneme_inventory_gold`, {PI.N_TOKENS} tokens, fingerprint '
      f'`{gold_fingerprint()}`), through the same `map_phoneme` the G2P uses. '
      f'Languages are BCP 47 codes. Two sources, never summed: `fleurs` is the '
      f'model\'s actual output in the `.gs.json` files (one vote per phoneme '
      f'token), `dict` is CharsiuG2P\'s training dictionaries (one vote per word '
      f'type).')
    A('')

    A('## 1. Sources')
    A('')
    fl = {c: m for c, m in langs.items() if m['fleurs']}
    if notes['paths_file']:
        A(f'`fleurs`: {sum(m["fleurs"]["files"] for m in fl.values()):,} files, '
          f'{len(fl)} languages, from `{os.path.basename(notes["paths_file"])}`.')
    else:
        A('`fleurs`: not used (no --paths given); every list is built from `dict` alone.')
    A('')
    if notes.get('missing_by_locale'):
        A('Locales with no `.gs.json` (g2p_task skips languages that need word '
          'segmentation; their groups rely on `dict` alone):')
        A('')
        A('| locale | clips without .gs.json |')
        A('|---|---:|')
        for loc, n in sorted(notes['missing_by_locale'].items()):
            A(f'| `{loc}` | {n} |')
        A('')
    if notes.get('mislabelled'):
        A('**Phonemized with the wrong regional variant.** These files were '
          'written from a paths_list that predates the BCP 47 switch: the clip '
          '`lang` held a bare ISO code, which g2p_task gives priority over the '
          'srt\'s own tag, so the regional variant was lost. They are counted '
          'under the variant actually used. Rebuild the paths_list with '
          '`fluers.make_paths_list` and rerun g2p for these locales.')
        A('')
        A('| locale | should be | tag used | files |')
        A('|---|---|---|---:|')
        for r in notes['mislabelled']:
            A(f'| `{r["locale"]}` | `{r["expected_lang"]}` (`{r["expected_tag"]}`) '
              f'| `{r["g2p_tag"]}` | {r["files"]} |')
        A('')
    stale = {c: m['fleurs']['stale_segments'] for c, m in fl.items()
             if m['fleurs']['stale_segments']}
    A(f'Consistency: re-mapping each file\'s `phonemes` reproduces its `gold_ph` '
      f'in every segment except {sum(stale.values())} '
      f'({", ".join(f"{c}: {n}" for c, n in stale.items()) or "none"}). A '
      f'mismatch means the file was written against a different inventory.')
    A('')

    A('## 2. Mapping health per language')
    A('')
    A('How each raw segment reached the inventory, on the primary source '
      '(`fleurs` where it exists, else `dict`). `unmapped` is what `.gs.json` '
      'lists under `gold_unmapped`; `backoff` is the larger and quieter loss. '
      f'Flagged (**bold**) at backoff >= {pct(FLAG_BACKOFF, 1, 0)} or unmapped >= '
      f'{pct(FLAG_UNMAPPED, 1)}.')
    A('')
    A('| lang | group | src | segments | direct | backoff | noise | unmapped | top backoff |')
    A('|---|---|---|---:|---:|---:|---:|---:|---|')
    for code, m in sorted(langs.items()):
        src, s = primary(m)
        if not s:
            continue
        n, o = s['segments'], s['outcome']
        flag = (o['backoff'] >= FLAG_BACKOFF * n or o['unmapped'] >= FLAG_UNMAPPED * n)
        name = f'**{code}**' if flag else code
        top = ', '.join(f'{md_code(k)}>{md_code(PI.BACKOFF[k])} {pct(v, n)}'
                        for k, v in list(s['backoff'].items())[:3])
        A(f'| {name} | {m["group"]} | {src} | {n:,} | {pct(o["direct"], n)} | '
          f'{pct(o["backoff"], n)} | {pct(o["noise"], n, 3)} | '
          f'{pct(o["unmapped"], n, 3)} | {top} |')
    A('')

    A('## 3. Inventory candidates')
    A('')
    A('Raw segments the gold inventory does not hold, ranked by their largest '
      'share of any one language\'s segments (primary source). A segment high '
      'here is either a phoneme the inventory is missing or a BACKOFF entry '
      'worth re-targeting -- e.g. a backoff that drops a contrast the language '
      'makes. Top 40, share >= 0.1%.')
    A('')
    cand = collections.defaultdict(lambda: {'max': 0.0, 'lang': '', 'n': 0, 'langs': set(),
                                            'kind': ''})
    for code, m in langs.items():
        src, s = primary(m)
        if not s or m['excluded']:
            continue
        for kind in ('backoff', 'unmapped', 'noise'):
            for seg, n in s[kind].items():
                c = cand[seg]
                c['n'] += n
                c['langs'].add(code)
                c['kind'] = kind
                share = n / s['segments']
                if share > c['max']:
                    c['max'], c['lang'] = share, code
    ranked = sorted(cand.items(), key=lambda kv: (-kv[1]['max'], kv[0]))
    A('| segment | now maps to | max share | in | total | languages |')
    A('|---|---|---:|---|---:|---|')
    for seg, c in [kv for kv in ranked if kv[1]['max'] >= 0.001][:40]:
        tgt = PI.BACKOFF.get(seg, PI.UNK)
        A(f'| {md_code(seg)} | {md_code(tgt)} | {pct(c["max"], 1)} | {c["lang"]} | {c["n"]:,} | '
          f'{" ".join(sorted(c["langs"]))} |')
    A('')

    A('## 4. Unmapped segments (`gold_unmapped`)')
    A('')
    A('Every segment that reached `<unk>`, both sources, all languages.')
    A('')
    A('| segment | fleurs | dict | languages |')
    A('|---|---:|---:|---|')
    um = collections.defaultdict(lambda: [0, 0, set()])
    for code, m in langs.items():
        for i, src in enumerate(('fleurs', 'dict')):
            s = m[src]
            for seg, n in (s['unmapped'] if s else {}).items():
                um[seg][i] += n
                um[seg][2].add(code)
    for seg, (a, b, ls) in sorted(um.items(), key=lambda kv: (-kv[1][0] - kv[1][1], kv[0])):
        A(f'| {md_code(seg)} | {a} | {b} | {" ".join(sorted(ls))} |')
    if not um:
        A('| - | 0 | 0 | |')
    A('')

    A('## 5. Group inventories')
    A('')
    A(f'Per language and source, the most frequent gold phonemes up to '
      f'{coverage:.1%} of its tokens; the group list is the union, in gold index '
      f'order, after the {len(PI.SPECIAL_TOKENS)} special tokens. `worst` is the '
      f'lowest coverage of any member language on either source. `dict only` '
      f'counts phonemes no FLEURS output asked for.')
    A('')
    A('| group | langs | tokens | worst | dict only | excluded |')
    A('|---|---:|---:|---|---:|---|')
    for g, r in inv.items():
        worst = min(((v, f'{c}:{s}') for c, d in r['coverage'].items()
                     for s, v in d.items() if v is not None), default=(None, ''))
        w = f'{pct(worst[0], 1)} {worst[1]}' if worst[0] is not None else '-'
        donly = sum(1 for v in r['selected_by'].values()
                    if not any(x.endswith(':fleurs') for x in v))
        A(f'| `{g}` | {len(r["langs"])} | {r["n_tokens"]} | {w} | {donly} | '
          f'{" ".join(r["excluded_langs"]) or "-"} |')
    A('')

    for g, r in inv.items():
        if not r['langs']:
            continue
        A(f'### `{g}`')
        A('')
        A(f'{r["description"]} Languages: {" ".join(r["langs"])}.')
        A('')
        gc = groups[g]
        fl_tot = sum(n for t, n in gc['fleurs']['gold'].items() if t not in PI.SPECIAL_TOKENS)
        A('`fleurs` share is of the group\'s pooled FLEURS output; `langs` is how '
          'many member languages emit the phoneme there. `selected by` names '
          'every language:source whose coverage needed it.')
        A('')
        A('| local | gold | phoneme | fleurs share | langs | selected by |')
        A('|---:|---:|---|---:|---:|---|')
        for i, t in enumerate(r['tokens']):
            if t in PI.SPECIAL_TOKENS:
                continue
            n = gc['fleurs']['gold'].get(t, 0)
            nl = gc['fleurs']['n_langs']['gold'].get(t, 0)
            by = r['selected_by'][t]
            by_s = ' '.join(by) if len(by) <= 6 else ' '.join(by[:6]) + f' +{len(by) - 6}'
            A(f'| {i} | {PI.TOKEN_INDEX[t]} | {md_code(t)} | {pct(n, fl_tot, 3)} | {nl} | {by_s} |')
        A('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def dump(obj, name):
    with open(os.path.join(OUT_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dicts', default=CHARSIU_DICTS)
    ap.add_argument('--paths', default=None,
                    help='FLEURS paths_list whose .gs.json files are counted; '
                         'omit to count the dictionaries only')
    ap.add_argument('--metadata-dir', default='',
                    help="directory the paths_list's '$/...' srt paths are relative to")
    ap.add_argument('--include-excluded', action='store_true',
                    help='also roll up languages in EXCLUDED_ISO')
    ap.add_argument('--coverage', type=float, default=COVERAGE,
                    help='per-language token coverage each group list must reach')
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()

    langs, notes = gather(args.dicts, paths_file=args.paths,
                          metadata_dir=args.metadata_dir, workers=args.workers)
    groups = rollup(langs, include_excluded=args.include_excluded)
    inv = select(langs, groups, coverage=args.coverage)

    gold = {'module': 'phoneme_inventory_gold', 'n_tokens': PI.N_TOKENS,
            'fingerprint': gold_fingerprint()}
    dump({'gold': gold, 'fleurs': notes,
          'langs': dict(sorted(langs.items()))}, 'lang_counts.json')
    dump(groups, 'group_counts.json')
    dump({'gold': gold,
          'coverage': args.coverage,
          'include_excluded': args.include_excluded,
          'special_tokens': list(PI.SPECIAL_TOKENS),
          # Every language the pipeline can emit, BCP 47 -> group, including
          # excluded ones, so a loader can route any .gs.json by its `lang`.
          'lang_to_group': {c: langs[c]['group'] for c in sorted(langs)},
          'aliases': lang_aliases(langs),
          'groups': inv}, 'group_inventories.json')
    write_report(langs, notes, groups, inv, args.coverage,
                 os.path.join(OUT_DIR, 'phoneme_counts.md'))

    print(f'wrote lang_counts.json, group_counts.json, group_inventories.json, '
          f'phoneme_counts.md to {OUT_DIR}', file=sys.stderr)
    for g, r in inv.items():
        print(f'  {g:18s} {len(r["langs"]):2d} langs  {r["n_tokens"]:3d} tokens',
              file=sys.stderr)


if __name__ == '__main__':
    main()

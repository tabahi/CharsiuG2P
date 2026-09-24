"""Per-language-group phoneme inventories: gold indices -> local indices.

The local tokens of a language group are the final output of this repo;
downstream tasks are expected to use them. Each language group's token list is
a SUBSET of the 270 gold tokens (see mappings/lang_group_inventories.json,
built by scripts/create_phoneme_inventories.py). The .gs.json files keep gold
indices only; the conversion happens here, at load time, so the lists can be
regenerated -- a different coverage target, a language added to EXCLUDED_ISO --
without rewriting any .gs.json.

"Language group" (lang_group) is never shortened to "group" in this repo:
"phoneme group" (phg, phoneme_features.GROUPS) is the other meaning.

    from standard_g2p import lang_group_inventory as LGI

    LGI.lang_group_of('pt-BR')                    # 'latin'
    t = LGI.to_local(seg['gold_ph'], 'pt-BR')     # dict: indices + their scope
    t['local_ph']                                 # indices into the language group's token list
    t['n_local_ph']                               # its token count (213)
    LGI.decode(t['local_ph'], 'pt-BR')            # ['p', 'a', 'ʁ', ...]

`lang` is a single BCP 47 code, as everywhere else. Other spellings lang_codes
accepts ('en-US', 'cmn', and the bare ISO 639-3 codes in .gs.json files written
before the BCP 47 switch) are resolved through the table's alias list.

Like lang_codes, this imports nothing heavy, so a data loader can use it
without pulling in transformers.
"""
import os
import json

try:                                  # imported as part of the package
    from . import phoneme_inventory_gold as PI
except ImportError:                   # imported with standard_g2p/ itself on sys.path
    import phoneme_inventory_gold as PI

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mappings', 'lang_group_inventories.json')

_TABLE = None


class StaleInventoryError(RuntimeError):
    """lang_group_inventories.json was built against a different gold inventory."""


def gold_fingerprint():
    """Same fingerprint create_phoneme_inventories.py records in the table."""
    import hashlib
    return hashlib.sha1('\n'.join(PI.TOKENS).encode('utf-8')).hexdigest()[:12]


def load(path=None):
    """The table, read once and checked against the current gold inventory.

    A language group's list indexes into gold TOKENS, so a list built against a
    different TOKENS maps every index to the wrong phoneme without any error.
    Refuse it here instead; the fix is to rerun
    scripts/create_phoneme_inventories.py.
    """
    global _TABLE
    if _TABLE is not None and path is None:
        return _TABLE
    with open(path or DEFAULT_PATH, encoding='utf-8') as f:
        table = json.load(f)
    want, got = gold_fingerprint(), table['gold']['fingerprint']
    if got != want:
        raise StaleInventoryError(
            f'lang_group_inventories.json was built against gold inventory {got}, '
            f'but phoneme_inventory_gold is now {want}. Rerun '
            f'scripts/create_phoneme_inventories.py.')
    if path is None:
        _TABLE = table
    return table


def canonical_lang(lang):
    """Any accepted spelling -> the canonical BCP 47 code the table is keyed by
    ('en-US' -> 'en', 'cmn' -> 'zh'). Raises KeyError on an unknown code."""
    t = load()
    code = t['aliases'].get(lang, lang)
    if code not in t['lang_to_lang_group']:
        raise KeyError(f'{lang!r} is not a language the G2P can emit; known: '
                       f'{sorted(t["lang_to_lang_group"])}')
    return code


def lang_group_of(lang):
    """BCP 47 code -> language group name.

    Excluded languages (lang_codes.EXCLUDED_ISO: 'my', 'nan', 'tts') still get
    their language group, but did not shape its list, so their coverage is not
    guaranteed -- 'burmese' has no usable language and holds only the specials.
    """
    return load()['lang_to_lang_group'][canonical_lang(lang)]


def inventory(lang_group):
    """The full record for one language group: tokens, gold_index,
    gold_to_local, coverage, selected_by, langs."""
    lang_groups = load()['lang_groups']
    try:
        return lang_groups[lang_group]
    except KeyError:
        raise KeyError(f'{lang_group!r} is not a language group; '
                       f'known: {sorted(lang_groups)}') from None


def n_tokens(lang_group):
    """Number of local tokens in a language group, specials included."""
    return inventory(lang_group)['n_tokens']


def tokens(lang_group):
    """Local index -> phoneme string. The 4 gold special tokens come first, at
    the same indices as in gold (0 = CTC blank)."""
    return inventory(lang_group)['tokens']


def to_local(gold_ph, lang):
    """Gold indices (a .gs.json `gold_ph`) -> the language group's local
    indices, with the scope needed to interpret them.

    Returns a dict:
        lang        canonical BCP 47 code the table is keyed by
        lang_group  the language group whose token list `local_ph` indexes
        local_ph    local index per gold index, same length as `gold_ph`
        n_local_ph  the language group's token count, specials included
        n_gold_ph   size of the gold space `gold_ph` indexed
        unk         the language group's <unk> index
        n_folded    how many entries of `gold_ph` were real phonemes outside
                    the language group's list and so became <unk> (a gold
                    <unk> that stays <unk> is not counted)
        gold_fingerprint  the gold inventory the table was built from

    A gold phoneme outside the language group's list becomes its <unk>. By
    construction that is under 0.1% of any member language's tokens, and
    phoneme_counts.md §5 has the exact figure per language.
    """
    code = canonical_lang(lang)
    lang_group = load()['lang_to_lang_group'][code]
    rec = inventory(lang_group)
    table = rec['gold_to_local']
    unk = rec['tokens'].index(PI.UNK)
    gold_unk = PI.TOKEN_INDEX[PI.UNK]
    local = [table[i] for i in gold_ph]
    return {
        'lang': code,
        'lang_group': lang_group,
        'local_ph': local,
        'n_local_ph': rec['n_tokens'],
        'n_gold_ph': PI.N_TOKENS,
        'unk': unk,
        'n_folded': sum(1 for g, l in zip(gold_ph, local) if l == unk and g != gold_unk),
        'gold_fingerprint': load()['gold']['fingerprint'],
    }


def decode(local, lang):
    """Local indices -> phoneme strings, for the language group `lang` belongs to."""
    toks = tokens(lang_group_of(lang))
    return [toks[i] for i in local]

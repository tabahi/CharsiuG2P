"""Per-group phoneme inventories: gold indices -> a group model's own indices.

Each processing group's model predicts a SUBSET of the 270 gold tokens (see
mappings/group_inventories.json, built by scripts/create_phoneme_inventories.py).
The .gs.json files keep gold indices only; the conversion happens here, at load
time, so the group lists can be regenerated -- a different coverage target, a
language added to EXCLUDED_ISO -- without rewriting any .gs.json.

    from standard_g2p import group_inventory as GI

    group = GI.group_of('pt-BR')                  # 'latin'
    t = GI.to_local(seg['gold_ph'], 'pt-BR')      # dict: indices + their scope
    t['local_ph']                                 # indices into that group's softmax
    t['n_local_ph']                               # the group's softmax size (213)
    GI.decode(t['local_ph'], 'pt-BR')             # ['p', 'a', 'ʁ', ...]

`lang` is a single BCP 47 code, as everywhere else. Other spellings lang_codes
accepts ('en-US', 'cmn', and the bare ISO 639-3 codes in .gs.json files written
before the BCP 47 switch) are resolved through the table's alias list.

Like lang_codes, this imports nothing heavy, so a training data loader can use it
without pulling in transformers.
"""
import os
import json

try:                                  # imported as part of the package
    from . import phoneme_inventory_gold as PI
except ImportError:                   # imported with standard_g2p/ itself on sys.path
    import phoneme_inventory_gold as PI

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mappings', 'group_inventories.json')

_TABLE = None


class StaleInventoryError(RuntimeError):
    """group_inventories.json was built against a different gold inventory."""


def gold_fingerprint():
    """Same fingerprint create_phoneme_inventories.py records in the table."""
    import hashlib
    return hashlib.sha1('\n'.join(PI.TOKENS).encode('utf-8')).hexdigest()[:12]


def load(path=None):
    """The table, read once and checked against the current gold inventory.

    A group list indexes into gold TOKENS, so a list built against a different
    TOKENS maps every index to the wrong phoneme without any error. Refuse it
    here instead; the fix is to rerun scripts/create_phoneme_inventories.py.
    """
    global _TABLE
    if _TABLE is not None and path is None:
        return _TABLE
    with open(path or DEFAULT_PATH, encoding='utf-8') as f:
        table = json.load(f)
    want, got = gold_fingerprint(), table['gold']['fingerprint']
    if got != want:
        raise StaleInventoryError(
            f'group_inventories.json was built against gold inventory {got}, '
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
    if code not in t['lang_to_group']:
        raise KeyError(f'{lang!r} is not a language the G2P can emit; known: '
                       f'{sorted(t["lang_to_group"])}')
    return code


def group_of(lang):
    """BCP 47 code -> processing group name.

    Excluded languages (lang_codes.EXCLUDED_ISO: 'my', 'nan', 'tts') still get
    their group, but did not shape its list, so their coverage is not
    guaranteed -- 'burmese' has no usable language and holds only the specials.
    """
    return load()['lang_to_group'][canonical_lang(lang)]


def inventory(group):
    """The full record for one group: tokens, gold_index, gold_to_local,
    coverage, selected_by, langs."""
    groups = load()['groups']
    try:
        return groups[group]
    except KeyError:
        raise KeyError(f'{group!r} is not a group; known: {sorted(groups)}') from None


def n_tokens(group):
    """Softmax size for a group's phoneme head, specials included."""
    return inventory(group)['n_tokens']


def tokens(group):
    """Local index -> phoneme string. The 4 gold special tokens come first, at
    the same indices as in gold (0 = CTC blank)."""
    return inventory(group)['tokens']


def to_local(gold_ph, lang):
    """Gold indices (a .gs.json `gold_ph`) -> the group's local indices, with
    the scope needed to interpret them.

    Returns a dict:
        lang        canonical BCP 47 code the table is keyed by
        group       processing group whose softmax `local_ph` indexes
        local_ph    local index per gold index, same length as `gold_ph`
        n_local_ph  the group's softmax size, specials included
        n_gold_ph   size of the gold space `gold_ph` indexed
        unk         the group's <unk> index
        n_folded    how many entries of `gold_ph` were real phonemes outside
                    the group's list and so became <unk> (a gold <unk> that
                    stays <unk> is not counted)
        gold_fingerprint  the gold inventory the group table was built from

    A gold phoneme outside the group's list becomes the group's <unk>. By
    construction that is under 0.1% of any member language's tokens, and
    phoneme_counts.md §5 has the exact figure per language.
    """
    code = canonical_lang(lang)
    group = load()['lang_to_group'][code]
    rec = inventory(group)
    table = rec['gold_to_local']
    unk = rec['tokens'].index(PI.UNK)
    gold_unk = PI.TOKEN_INDEX[PI.UNK]
    local = [table[i] for i in gold_ph]
    return {
        'lang': code,
        'group': group,
        'local_ph': local,
        'n_local_ph': rec['n_tokens'],
        'n_gold_ph': PI.N_TOKENS,
        'unk': unk,
        'n_folded': sum(1 for g, l in zip(gold_ph, local) if l == unk and g != gold_unk),
        'gold_fingerprint': load()['gold']['fingerprint'],
    }


def decode(local, lang):
    """Local indices -> phoneme strings, for the group `lang` belongs to."""
    toks = tokens(group_of(lang))
    return [toks[i] for i in local]

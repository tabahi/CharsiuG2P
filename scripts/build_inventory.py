"""Build the gold phoneme inventory + backoff map, emit phoneme_inventory_gold.py.

Step 2 of rebuilding the gold inventory; run scripts/layers.py first.

    python scripts/build_inventory.py
"""
import os, sys, json, collections, unicodedata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'standard_g2p'))
from gold_g2p import strip_to_base, TIE_BARS

D = os.path.join(ROOT, 'tmp', 'inventory_build')
d = json.load(open(os.path.join(D, 'layers.json')))
seg = collections.Counter(dict(d['segments']))
seg_langs = {k: set(v) for k, v in d['seg_langs'].items()}

MIN_LANGS = 2
inv = [s for s, _ in seg.most_common() if len(seg_langs[s]) >= MIN_LANGS]
inv_set = set(inv)
excluded = [s for s, _ in seg.most_common() if s not in inv_set]
print('inventory: %d   excluded: %d' % (len(inv), len(excluded)))

# Diacritics ordered by how much phonemic weight they usually carry, least
# first. Backoff peels them in this order until we land on something in the
# inventory, so the most detail is dropped before anything contrastive.
TIERS = [
    # tier 1: fine phonetic detail, almost never contrastive
    '̠̟̝̞̥̬̪̽̈̊̚'
    '̺̻̹̜̘̙˔˕͈͇',
    # tier 2: phonation / syllabicity / nasality
    '̴̴̯̩̰̤̼͓͖̃',
    # tier 3: secondary articulation and release
    'ʲʷˠˤʰʱⁿˡʼˀ˞ᵊ',
]


def peel(tok, chars):
    nfd = unicodedata.normalize('NFD', tok)
    out = ''.join(c for c in nfd if c not in chars)
    return unicodedata.normalize('NFC', out)


# Segments the automatic peel cannot reach, because the target differs by more
# than a diacritic. Implosives are the important case: they are frequent in
# Swahili here and widespread in African and SE-Asian languages generally, but
# only one corpus language attests them, so the >=2-language rule drops them.
MANUAL = {
    'ɓ': 'b', 'ɗ': 'd', 'ʄ': 'ɟ', 'ɠ': 'ɡ', 'ʛ': 'ɡ',   # implosives -> plain voiced stop
    'ɢ': 'ɡ',                                            # voiced uvular -> velar
    'ɶ': 'œ',                                            # open front rounded
    'ɞ': 'ɔ',                                            # open-mid central rounded
    'ɮ': 'l',                                            # voiced lateral fricative
    'ʙ': 'b',                                            # bilabial trill
    'ᵻ': 'ɨ',
}

# Not phonetic at all: kana leaking from jpn, Arabic letters left untranscribed
# in fas/kur, control characters, punctuation, digits. These are transcription
# failures and should be routed to NOISE, not to a phoneme.
def _is_junk(tok):
    for ch in tok:
        cat = unicodedata.category(ch)
        if cat in ('Cf', 'Cc', 'Cn', 'Po', 'Nd', 'Pc', 'Pe', 'Ps'):
            return True
        o = ord(ch)
        if 0x3000 <= o <= 0x30ff or 0x0600 <= o <= 0x06ff or ch == '�':
            return True
    return False


def backoff_for(tok):
    """Best in-inventory target for an out-of-inventory segment."""
    if tok in MANUAL:
        return MANUAL[tok], 'manual'
    if _is_junk(tok):
        return '\x00NOISE', 'junk'
    # a manual target may still need a diacritic peeled: ɓʲ -> bʲ
    base = strip_to_base(tok)
    if base in MANUAL and MANUAL[base] in inv_set:
        rebuilt = tok.replace(base, MANUAL[base], 1)
        if rebuilt in inv_set:
            return rebuilt, 'manual'
        return MANUAL[base], 'manual'
    # progressively strip tiers
    acc = ''
    for tier in TIERS:
        acc += tier
        cand = peel(tok, acc)
        if cand and cand in inv_set:
            return cand, 'peel'
    # drop everything -> bare base
    base = strip_to_base(tok)
    if base in inv_set:
        return base, 'base'
    # tie-bar unit with no single-unit target: split it
    nfd = unicodedata.normalize('NFD', tok)
    if any(c in TIE_BARS for c in nfd):
        parts = [p for p in strip_to_base(tok)]
        if all(p in inv_set for p in parts) and parts:
            return ' '.join(parts), 'split'
        # try first element alone
        if parts and parts[0] in inv_set:
            return parts[0], 'head'
    # last resort: first base letter
    if base and base[0] in inv_set:
        return base[0], 'first'
    return None, 'unk'


backoff = {}
stats = collections.Counter()
lost = []
for s in excluded:
    tgt, how = backoff_for(s)
    stats[how] += 1
    if tgt is None:
        lost.append((s, seg[s]))
    else:
        backoff[s] = tgt

print('backoff resolution:', dict(stats))
print('unresolvable: %d types, %d occurrences' % (len(lost), sum(n for _, n in lost)))
print('  examples:', [s for s, _ in lost[:15]])

tot = sum(seg.values())
cov_direct = sum(seg[s] for s in inv) / tot
cov_backed = (sum(seg[s] for s in inv) + sum(seg[s] for s in backoff)) / tot
print('coverage: direct %.4f%%   with backoff %.4f%%' % (100 * cov_direct, 100 * cov_backed))

# ---- emit module ----
lines = []
lines.append('"""Standard multilingual phoneme inventory (generated - do not edit by hand).')
lines.append('')
lines.append('Built from the CharsiuG2P training corpus (%d pronunciations, 100 languages)' % 7721984)
lines.append('by `scripts/build_inventory.py`. A segment is admitted when it is attested in')
lines.append('at least %d languages -- multi-language evidence is what makes a class learnable' % MIN_LANGS)
lines.append('as a universal unit rather than a memorised language-specific quirk.')
lines.append('')
lines.append('    inventory      %d segments' % len(inv))
lines.append('    direct cover   %.3f%% of corpus tokens' % (100 * cov_direct))
lines.append('    with backoff   %.3f%%' % (100 * cov_backed))
lines.append('')
lines.append('Suprasegmentals (tone, stress, length) are separate layers - see gold_g2p.py.')
lines.append('')
lines.append('    from standard_g2p import phoneme_inventory_gold as PI')
lines.append('')
lines.append('    PI.TOKENS[i]          # index -> phoneme string')
lines.append('    PI.TOKEN_INDEX[p]     # phoneme string -> index')
lines.append('    PI.decode(ph)         # whole sequence at once')
lines.append('    PI.N_TOKENS           # %d, the gold inventory size' % (len(inv) + 4))
lines.append('"""')
lines.append('')
lines.append('MIN_LANGUAGES = %d' % MIN_LANGS)
lines.append('')
lines.append('# Special tokens occupy the low indices; phonemes follow contiguously.')
lines.append("BLANK, SIL, NOISE, UNK = '<blank>', 'SIL', 'noise', '<unk>'")
lines.append('SPECIAL_TOKENS = [BLANK, SIL, NOISE, UNK]')
lines.append('')
lines.append('# Ordered by corpus frequency, descending.')
lines.append('PHONEMES = [')
for i in range(0, len(inv), 8):
    lines.append('    ' + ' '.join("'%s'," % p for p in inv[i:i + 8]))
lines.append(']')
lines.append('')
lines.append('# segment -> (occurrences, n_languages), kept so the threshold can be')
lines.append('# revisited without recomputing from the corpus.')
lines.append('PHONEME_STATS = {')
for p in inv:
    lines.append("    '%s': (%d, %d)," % (p, seg[p], len(seg_langs[p])))
lines.append('}')
lines.append('')
lines.append('# Out-of-inventory segment -> in-inventory target. A space-separated value')
lines.append('# means the segment expands to several units.')
lines.append('BACKOFF = {')
for s in sorted(backoff, key=lambda x: -seg[x]):
    tgt = backoff[s]
    lines.append("    %r: %s," % (s, 'NOISE' if tgt == '\x00NOISE' else repr(tgt)))
lines.append('}')
lines.append('')
lines.append('''
TOKENS = SPECIAL_TOKENS + PHONEMES
TOKEN_INDEX = {t: i for i, t in enumerate(TOKENS)}
PHONEME_INDEX = TOKEN_INDEX
N_TOKENS = len(TOKENS)


def map_phoneme(seg):
    """Map one segment onto the inventory. Returns a list (a segment may expand
    to several units, or to [] if nothing sensible remains)."""
    if seg in TOKEN_INDEX:
        return [seg]
    tgt = BACKOFF.get(seg)
    if tgt is None:
        return [UNK]
    return tgt.split(' ')


def map_sequence(segments, drop_unk=False):
    """Map a segment sequence onto inventory strings."""
    out = []
    for s in segments:
        for m in map_phoneme(s):
            if drop_unk and m == UNK:
                continue
            out.append(m)
    return out


def encode(segments, drop_unk=False):
    """Map a segment sequence onto integer indices."""
    return [TOKEN_INDEX[t] for t in map_sequence(segments, drop_unk=drop_unk)]


def decode(indices):
    return [TOKENS[i] for i in indices]
'''.strip())
lines.append('')

out = os.path.join(ROOT, 'standard_g2p', 'phoneme_inventory_gold.py')
open(out, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', out)
json.dump({'inventory': inv, 'backoff': backoff,
           'stats': {p: [seg[p], len(seg_langs[p])] for p in inv}},
          open(os.path.join(D, 'inventory.json'), 'w'), ensure_ascii=False)

"""Generate the articulatory feature table for the gold inventory.

Step 3 of rebuilding the gold inventory; needs panphon. Writes
standard_g2p/phoneme_features.py.

    python scripts/build_features.py
"""
import os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'standard_g2p'))
import panphon
import phoneme_inventory_gold as PI
from gold_g2p import strip_to_base, TIE_BARS
import unicodedata

ft = panphon.FeatureTable()
ALL = ft.names

# panphon spells r-coloured vowels with the rhotic hook modifier.
ALIASES = {'ɚ': 'ə˞', 'ɝ': 'ɜ˞'}

# Features that carry no information here: length and tone are separate layers,
# and no click survived the inventory threshold. Dropping them keeps the vector
# honest about what it actually distinguishes.
DROP = {'long', 'hitone', 'hireg', 'velaric'}
NAMES = [n for n in ALL if n not in DROP]
KEEP_IDX = [i for i, n in enumerate(ALL) if n not in DROP]


# panphon applies most diacritics compositionally (ʰ->sg, ̃->nas, ɓ->cg, ̪->distr,
# ˤ->lo, ʷ->round, ̥->voi, ʼ->cg) but SILENTLY DROPS a few. Without these patches
# every palatalised consonant collapses onto its plain counterpart, which would
# erase a contrast that is phonemic across Russian, Polish, Lithuanian and Irish.
MODIFIER_PATCHES = {
    'ʲ': {'hi': 1, 'back': -1},    # palatalised: secondary dorsal at the palate
    'ʱ': {'sg': 1, 'voi': 1},      # breathy/murmured release
}
# Deliberately NOT patched. These mark sub-phonemic detail with no correlate in
# a 20-feature system, so letting them share their base's vector is correct:
#   ̠ retracted   ̝ raised   ̞ lowered   ̈ centralized   ̚ unreleased   ̺ apical
IGNORED_BY_DESIGN = '̠̝̞̺̻͇͈̈̚'


def _apply_patches(vec, tok):
    vec = list(vec)
    for mod, patch in MODIFIER_PATCHES.items():
        if mod in tok:
            for name, val in patch.items():
                vec[ALL.index(name)] = val
    return vec


def raw_vectors(p):
    vs = ft.word_to_vector_list(ALIASES.get(p, p), numeric=True)
    if any(m in p for m in MODIFIER_PATCHES):
        vs = [_apply_patches(v, p) for v in vs]
    return vs


def parts_of(tok):
    """Base letters either side of a tie bar."""
    return list(strip_to_base(tok))


def synth_affricate(tok):
    """A tie-bar affricate panphon does not know as a unit (t͡ʂ, d͡ʐ).

    Place comes from the fricative, which is the second element; the result is
    a stop-like [-continuant] with [+delayed release]. Voicing follows the
    fricative too, since both halves agree in a well-formed affricate.
    """
    ps = parts_of(tok)
    if len(ps) != 2:
        return None
    v = raw_vectors(ps[1])
    if not v:
        return None
    v = list(v[0])
    v[ALL.index('cont')] = -1
    v[ALL.index('delrel')] = 1
    return v


VOWELS = set('iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ')

features = {}
complex_units = {}
notes = collections.Counter()

# Vectors are computed for the excluded segments too, not just the inventory:
# `nearest()` is meant to place an OUT-of-inventory segment, so it needs that
# segment's own vector to compare against.
ALL_SEGMENTS = list(PI.PHONEMES) + [s for s in PI.BACKOFF if s not in set(PI.PHONEMES)]

for p in ALL_SEGMENTS:
    vs = raw_vectors(p)
    if len(vs) == 1:
        features[p] = vs[0]
        notes['direct'] += 1
        continue

    ps = parts_of(p)
    is_tie = any(c in TIE_BARS for c in unicodedata.normalize('NFD', p))
    if len(vs) == 0:
        notes['missing'] += 1
        features[p] = None
        continue

    # more than one vector: either an affricate panphon lacks, or a genuinely
    # two-target unit (diphthong, co-articulated stop).
    if is_tie and len(ps) == 2 and ps[0] not in VOWELS and ps[1] not in VOWELS:
        v = synth_affricate(p)
        if v:
            features[p] = v
            notes['synth_affricate'] += 1
            continue
    # Genuine two-target unit: keep the first vector as the primary and record
    # that it is complex, rather than averaging into something that describes
    # neither half.
    features[p] = vs[0]
    complex_units[p] = len(vs)
    notes['complex_first'] += 1

print('resolution:', dict(notes))
print('complex units:', complex_units)
missing = [p for p, v in features.items() if v is None]
print('missing:', missing)

# --- validate the synthesis rule against a case panphon DOES know ---
known = ft.word_to_vector_list('t͡ʃ', numeric=True)[0]
synth = synth_affricate('t͡ʃ')
diff = [ALL[i] for i in range(len(ALL)) if known[i] != synth[i]]
print('\nsynthesis check on t͡ʃ (panphon vs synthesised): differing features = %s' % (diff or 'none'))

# --- trim to informative features ---
trimmed = {p: tuple(v[i] for i in KEEP_IDX) for p, v in features.items() if v is not None}
print('\nfeatures kept: %d of %d  -> %s' % (len(NAMES), len(ALL), NAMES))

# --- duplicate detection: phonemes a feature decoder cannot separate ---
byvec = collections.defaultdict(list)
_inv_set=set(PI.PHONEMES)
for p, v in trimmed.items():
    if p in _inv_set: byvec[v].append(p)
dups = {v: ps for v, ps in byvec.items() if len(ps) > 1}
print('\ncollision groups (identical feature vectors): %d' % len(dups))

ACCENT = 'áàâǎăéèêěóòôǒíìîǐúùûǔýỳŷ'
def why(p, head):
    nfd = unicodedata.normalize('NFD', p)
    if any(c in unicodedata.normalize('NFD', ACCENT) and unicodedata.category(c) == 'Mn'
           for c in nfd):
        return 'tone/pitch-accent'
    if any(c in IGNORED_BY_DESIGN for c in nfd):
        return 'sub-phonemic detail'
    if any(c in TIE_BARS for c in nfd):
        return 'diphthong (1st target only)'
    return 'UNEXPLAINED'

cat = collections.Counter()
unexplained = []
for v, ps in dups.items():
    head = ps[0]
    for p in ps[1:]:
        r = why(p, head)
        cat[r] += 1
        if r == 'UNEXPLAINED':
            unexplained.append((head, p))
print('collapsed phonemes by cause:', dict(cat))
print('phonemes involved in a collision: %d / %d'
      % (sum(len(p) for p in dups.values()), len(trimmed)))
if unexplained:
    print('UNEXPLAINED collisions (%d):' % len(unexplained), unexplained[:25])

# Emitted verbatim into the generated module. Kept in a template file because it
# contains its own docstrings and cannot be nested in a string literal here.
GROUPS_BLOCK = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'groups_block.py.in'),
                    encoding='utf-8').read().strip()


# ---------------- emit ----------------
L = []
L.append('"""Articulatory feature vectors for the standard inventory (generated).')
L.append('')
L.append('Built by `scripts/build_features.py` from panphon %s. Each phoneme maps to' % panphon.__version__ if hasattr(panphon, '__version__') else 'Built by `scripts/build_features.py` from panphon. Each phoneme maps to')
L.append('%d ternary features (+1 present, -1 absent, 0 not applicable).' % len(NAMES))
L.append('')
L.append('Four of panphon\'s 24 features are dropped because they carry no information')
L.append('here: `long`, `hitone` and `hireg` are handled as separate layers (see')
L.append('gold_g2p.decompose_ipa), and `velaric` is constant since no click met the')
L.append('inventory threshold.')
L.append('')
L.append('The point of this table is compositional generalisation: an unseen phoneme')
L.append('still has a feature vector, so it can be placed by `nearest()` rather than')
L.append('falling off the inventory. Implosive b -> b differs from plain b only in')
L.append('`cg`, which the hand-written BACKOFF in phoneme_inventory_gold.py could not know.')
L.append('"""')
L.append('')
# Works both as a package member and as a plain script on sys.path.
L.append('try:')
L.append('    from . import phoneme_inventory_gold as _inv')
L.append('except ImportError:')
L.append('    import phoneme_inventory_gold as _inv')
L.append('')
L.append('FEATURE_NAMES = %r' % (NAMES,))
L.append('N_FEATURES = len(FEATURE_NAMES)')
L.append('')
L.append('# Units that are really two acoustic targets (diphthongs, co-articulated')
L.append('# stops). The stored vector describes the FIRST target only.')
L.append('COMPLEX_UNITS = %r' % (complex_units,))
L.append('')
L.append('# Inventory phonemes.')
L.append('FEATURES = {')
for p in PI.PHONEMES:
    if p in trimmed:
        L.append('    %r: %r,' % (p, trimmed[p]))
L.append('}')
L.append('')
L.append('# Segments that did NOT make the inventory. Kept so `nearest()` can place')
L.append('# them: it needs the query segment\'s own vector, not just the targets\'.')
L.append('EXTRA_FEATURES = {')
for p in ALL_SEGMENTS:
    if p not in set(PI.PHONEMES) and p in trimmed:
        L.append('    %r: %r,' % (p, trimmed[p]))
L.append('}')
L.append('')
L.append('''
_ZERO = (0,) * N_FEATURES
_ALL = dict(FEATURES)
_ALL.update(EXTRA_FEATURES)


def feature_vector(phoneme, use_panphon=True):
    """Feature tuple for a phoneme.

    Covers inventory phonemes and every segment seen in the source corpus. For
    anything else, panphon is consulted at runtime when available (import is
    lazy, so panphon is not a hard dependency). Special/unknown -> all zeros.
    """
    # Special tokens are NAMES, not IPA, and must be caught before the panphon
    # fallback: it happily parses '<blank>' as b,l,a,n,k and hands back the
    # vector for /b/, which would train a feature head on nonsense.
    if phoneme in _inv.SPECIAL_TOKENS:
        return _ZERO
    v = _ALL.get(phoneme)
    if v is not None:
        return v
    if use_panphon:
        v = _panphon_vector(phoneme)
        if v is not None:
            _ALL[phoneme] = v
            return v
    return _ZERO


_FT = None


def _panphon_vector(phoneme):
    """Last-resort runtime lookup for a segment absent from the baked tables."""
    global _FT
    try:
        if _FT is None:
            import panphon
            _FT = panphon.FeatureTable()
        vs = _FT.word_to_vector_list(phoneme, numeric=True)
        if not vs:
            return None
        keep = [i for i, n in enumerate(_FT.names) if n in FEATURE_NAMES]
        return tuple(vs[0][i] for i in keep)
    except Exception:
        return None


def feature_matrix(tokens=None):
    """Rows aligned with `tokens` (default: the full token list, so row i is
    the vector for index i). Feed straight to a feature-prediction head."""
    tokens = tokens if tokens is not None else _inv.TOKENS
    return [feature_vector(t) for t in tokens]


def distance(a, b):
    """Count of differing features between two phonemes."""
    va, vb = feature_vector(a), feature_vector(b)
    return sum(1 for x, y in zip(va, vb) if x != y)


def nearest(phoneme, k=1, candidates=None):
    """Closest INVENTORY phoneme(s) to `phoneme` by feature distance.

    This is the feature-space backoff: unlike the hand-written BACKOFF table in
    phoneme_inventory_gold.py it is derived from phonetics, so it places segments
    nobody enumerated. Ties are broken by inventory frequency (PHONEMES is
    ordered by corpus count), so the commoner phoneme wins.
    """
    cands = candidates if candidates is not None else _inv.PHONEMES
    v = feature_vector(phoneme)
    if v == _ZERO:
        return []
    rank = {p: i for i, p in enumerate(_inv.PHONEMES)}
    scored = sorted(((sum(1 for x, y in zip(v, FEATURES[c]) if x != y),
                      rank.get(c, 1 << 30), c)
                     for c in cands if c != phoneme and c in FEATURES))
    return [c for _, _, c in scored[:k]]


def feature_targets(tokens=None):
    """Binary targets plus an applicability mask, rows aligned with `tokens`.

        targets[i][j]   1.0 if feature j is present on token i, else 0.0
        mask[i][j]      1.0 if feature j APPLIES to token i, else 0.0

    The stored features are ternary (+1 present, -1 absent, 0 not applicable),
    which does not fit a binary head: `distr` is undefined on a vowel, `tense`
    on a consonant, and roughly a tenth of all cells are such holes. Taking a
    plain BCE over them trains the head toward a meaningless middle value, so
    the mask marks where the loss should actually be taken.

    Special tokens (blank, SIL, noise, unk) come back fully masked, so no
    feature loss is charged on those frames at all.

        feat, mask = feature_targets()
        loss = (bce(logits, feat) * mask).sum() / mask.sum().clamp(min=1)
    """
    rows = feature_matrix(tokens)
    targets = [[1.0 if v == 1 else 0.0 for v in row] for row in rows]
    mask = [[0.0 if v == 0 else 1.0 for v in row] for row in rows]
    return targets, mask


def save_npz(path):
    """Dump the tables so a training repo can use them without importing this
    module (and without installing panphon). numpy is imported lazily."""
    import numpy as np
    targets, mask = feature_targets()
    np.savez(
        path,
        tokens=np.array(_inv.TOKENS, dtype=object),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        features=np.array(feature_matrix(), dtype=np.int8),
        targets=np.array(targets, dtype=np.float32),
        mask=np.array(mask, dtype=np.float32),
        group_of_token=np.array([GROUPS.get(t, 0) for t in _inv.TOKENS],
                                dtype=np.int16),
        group_names=np.array(GROUP_NAMES, dtype=object),
    )
    return path
'''.strip())
L.append('')
L.append(GROUPS_BLOCK)
L.append('')

out = os.path.join(ROOT, 'standard_g2p', 'phoneme_features.py')
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print('\nwrote', out, '(%d phonemes with vectors)' % len(trimmed))

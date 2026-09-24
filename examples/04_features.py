"""Articulatory features for an auxiliary training head. No model needed.

    python examples/04_features.py

Features are a lookup keyed by gold index and are never stored per file: the
vector is a deterministic function of the phoneme.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standard_g2p import phoneme_inventory_gold as PI
from standard_g2p import phoneme_features as PF

for p in ['p', 'b', 'pʰ', 't͡ʃ', 'a', 'i', 'ŋ']:
    on = [n for n, x in zip(PF.FEATURE_NAMES, PF.feature_vector(p)) if x == 1]
    print('%-4s %-14s %s' % (p, PF.GROUP_NAMES[PF.GROUPS[p]], ' '.join(on)))

# Minimal pairs differ in exactly the features that separate them.
print()
for a, b in [('p', 'b'), ('p', 'pʰ'), ('b', 'ɓ'), ('s', 'ʃ'), ('i', 'u')]:
    diff = [n for n, x, y in zip(PF.FEATURE_NAMES, PF.feature_vector(a),
                                 PF.feature_vector(b)) if x != y]
    print('%-3s vs %-3s distance=%d %s' % (a, b, PF.distance(a, b), diff))

# Features place a segment the inventory never admitted. `nearest()` is not
# wired into the mapping; BACKOFF is what the pipeline actually uses.
print('\n%-6s %-8s %s' % ('seg', 'BACKOFF', 'nearest() by features'))
for p in ['ɓ', 'ɗ', 'ʄ', 'ʈ͡ʂ', 'ɮ']:
    print('%-6s %-8s %s' % (p, PI.BACKOFF.get(p, '-'), PF.nearest(p, k=3)))

# For training: binary targets plus a mask of where each feature applies.
# Stored values are ternary and 0 means "not applicable" (distr on a vowel),
# so a plain BCE over them would train toward a meaningless middle value.
feat, mask = PF.feature_targets()           # rows aligned with gold TOKENS
print('\nfeature_targets(): %d x %d targets and mask' % (len(feat), len(feat[0])))
assert all(sum(mask[i]) == 0 for i in range(len(PI.SPECIAL_TOKENS)))
print("""
    FEAT, MASK = torch.tensor(feat), torch.tensor(mask)   # (270, 20)
    tgt, m = FEAT[gold_ph], MASK[gold_ph]                 # (B, T) -> (B, T, 20)
    aux = (bce(feat_logits, tgt) * m).sum() / m.sum().clamp(min=1)

PF.save_npz(path) writes the same tables for a repo that cannot import this one.""")

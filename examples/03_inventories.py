"""The gold inventory and the per-group inventories. No model needed.

    python examples/03_inventories.py

Shows what to size each model head to, and how one gold phoneme lands at a
different local index in each group model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standard_g2p import phoneme_inventory_gold as PI
from standard_g2p import phoneme_features as PF
from standard_g2p import group_inventory as GI
from standard_g2p.gold_g2p import TONE_VOCAB, N_TONES, N_STRESS, N_LENGTHS

print('gold phoneme head : %3d  (%d special + %d phonemes)'
      % (PI.N_TOKENS, len(PI.SPECIAL_TOKENS), len(PI.PHONEMES)))
print('broad group head  : %3d  %s' % (PF.N_GROUPS, PF.GROUP_NAMES))
print('tone head         : %3d  %s' % (N_TONES, TONE_VOCAB))
print('stress head       : %3d  none / primary / secondary' % N_STRESS)
print('length head       : %3d  short / half-long / long' % N_LENGTHS)
print('feature head      : %3d  %s' % (PF.N_FEATURES, PF.FEATURE_NAMES))

# `gold_ph` indexes TOKENS, not PHONEMES. TOKENS = SPECIAL_TOKENS + PHONEMES,
# so indexing PHONEMES with a gold index is silently off by four.
print('\nTOKENS[:6]   = %s' % PI.TOKENS[:6])
print('PHONEMES[:2] = %s  (PHONEMES[0] is gold index %d)'
      % (PI.PHONEMES[:2], PI.TOKEN_INDEX[PI.PHONEMES[0]]))

table = GI.load()        # raises StaleInventoryError if built against another gold
print('\ngroup tables: gold fingerprint %s, per-language coverage target %.1f%%'
      % (table['gold']['fingerprint'], 100 * table['coverage']))
print('%-18s %6s  %s' % ('group', 'tokens', 'languages (BCP 47)'))
for g, r in table['groups'].items():
    langs = ' '.join(r['langs'][:12]) + (' ...' if len(r['langs']) > 12 else '')
    print('%-18s %6d  %s' % (g, r['n_tokens'], langs or '(none usable)'))
    # Specials keep their gold index in every group: CTC blank is 0 everywhere.
    assert GI.tokens(g)[:len(PI.SPECIAL_TOKENS)] == PI.SPECIAL_TOKENS

# Any spelling lang_codes accepts routes, not only the canonical one.
print('\nrouting:')
for lang in ['en-US', 'en-GB', 'pt-BR', 'pt', 'cmn', 'zh-Hant', 'hr']:
    print('   %-8s -> %-6s %s' % (lang, GI.canonical_lang(lang), GI.group_of(lang)))

# One phoneme, one gold index, a different local index per group -- and the
# group's <unk> (3) where none of its languages use it.
gi = PI.TOKEN_INDEX['kʰ']
print("\n'kʰ' is gold %d; local index per group:" % gi)
for g, r in table['groups'].items():
    if r['langs']:
        print('   %-18s %d' % (g, r['gold_to_local'][gi]))

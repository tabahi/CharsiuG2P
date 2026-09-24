"""Sentence -> standardized phoneme layers -> a group model's training targets.

    python examples/01_phonemize_sentence.py

This is the whole path a training loader follows, except that a loader reads
`gold_ph` back from a .gs.json instead of running the model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from standard_g2p.gold_g2p import goldG2P
from standard_g2p import group_inventory as GI
from standard_g2p import phoneme_inventory_gold as PI

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
G = goldG2P(device=device)

# 1. One sentence, in full. `lang` is always a single BCP 47 code; 'pt-BR' and
#    bare 'pt' (European) are different models of the language.
d = G.phonemize_sentence('uma parceria', lang='pt-BR')
for k, v in d.items():
    print('%-14s %s' % (k, v))

# Every index comes with the size of the space it indexes.
assert all(0 <= t < d['n_tones'] for t in d['tone'])
assert all(0 <= s < d['n_stresses'] for s in d['stress'])
assert all(0 <= n < d['n_lengths'] for n in d['length'])
assert all(0 <= i < d['n_gold_ph'] for i in d['gold_ph'])
assert all(0 <= g < d['n_gold_phg'] for g in d['gold_phg'])

# 2. Gold indices -> the indices of the language's group model.
t = GI.to_local(d['gold_ph'], 'pt-BR')
print()
for k, v in t.items():
    print('%-16s %s' % (k, v))
print('decoded          %s' % GI.decode(t['local_ph'], 'pt-BR'))

# 3. The same, across scripts and groups.
print()
for lang, text in [('en-US', 'hello world'),
                   ('ru', 'привет мир'),
                   ('hi', 'नमस्ते दुनिया'),
                   ('ko', '안녕하세요'),
                   ('th', 'ภาษา'),        # tonal; one word, Thai needs segmenting
                   ('zh', '狂妄')]:        # tonal; one word, Chinese needs segmenting
    d = G.phonemize_sentence(text, lang=lang)
    t = GI.to_local(d['gold_ph'], lang)
    print('%-6s %-14s group=%-16s local %d-way, gold %d-way'
          % (lang, text, t['group'], t['n_local_ph'], t['n_gold_ph']))
    print('       phonemes %s' % d['phonemes'])
    print('       tone     %s' % d['tone'])
    print('       decoded  %s' % GI.decode(t['local_ph'], lang))

    # Round trip: a gold phoneme on the group's list decodes to itself,
    # anything else to <unk> -- and n_folded counts exactly those.
    keep = set(GI.tokens(t['group']))
    want = [p if p in keep else PI.UNK for p in PI.decode(d['gold_ph'])]
    assert GI.decode(t['local_ph'], lang) == want
    if t['n_folded']:
        print('       %d phoneme(s) not on the %s list -> <unk>' % (t['n_folded'], t['group']))

print()
G.print_stats()

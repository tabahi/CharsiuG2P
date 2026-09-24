"""How a raw IPA string is standardized and split into layers. No model needed.

    python examples/02_ipa_layers.py

Each input below is a real artifact of the CharsiuG2P training data that the
model reproduces at inference.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standard_g2p.gold_g2p import (normalize_ipa, decompose_ipa, normalize_tone,
                                   TONE_INDEX, split_words)
from standard_g2p import phoneme_inventory_gold as PI

for tag, ipa, what in [
        ('tha',   'pʰaː˧.saː˩˩˦',    'tone letters -> tone layer, ː -> length layer'),
        ('nan',   'kʰuan²¹⁻⁵³to²²', 'Chao digits, not tone letters'),
        ('swe',   'plA:na%vE:gen',  'SAMPA, not IPA'),
        ('ara',   'qatˤˤala',       'modifier typed twice'),
        ('arm-e', 'tʰəɾtʰənd͡ʒuk',   'tie-bar affricate is one segment'),
        ('eng-us', 'ˈʧɑɹ',          'stress mark -> stress layer; ʧ ligature -> t͡ʃ'),
        ('kor',   't͈ɑŋ',            'segment outside the inventory -> BACKOFF')]:
    d = decompose_ipa(ipa, lang=tag)
    print('%-6s %-16s %s' % (tag, ipa, what))
    print('   normalized %s' % normalize_ipa(ipa, lang=tag))
    # Every layer has one entry per segment. Tone sits on the syllable nucleus,
    # so consonants carry '' (no tone) -- a blank, not a missing entry.
    print('   segments   %s' % d['segments'])
    print('   tone       %s -> index %s'
          % ([normalize_tone(t) or '-' for t in d['tone']],
             [TONE_INDEX.get(normalize_tone(t), 0) for t in d['tone']]))
    print('   stress     %s' % d['stress'])
    print('   length     %s' % d['length'])
    print('   gold       %s' % PI.map_sequence(d['segments']))
    print()

# The word splitter keeps combining marks inside words (Brahmic vowel signs,
# viramas), which a `\w`-based regex would treat as separators.
print(split_words('Char siu is a Cantonese style.'))
print(split_words('புஷ்ஷின் நிர்வாகம்'))

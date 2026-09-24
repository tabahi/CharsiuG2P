"""Word segmentation for scripts without spaces.

    python examples/06_word_segmentation.py

CharsiuG2P takes one word at a time. Thai, Khmer, Burmese, Japanese and Chinese
do not put spaces between words, so `split_words` would hand the model whole
clauses. `segment` splits them into words first (see
standard_g2p/word_segmentation.py for the backend per language and why).
Part 1 needs only the segmenter packages; part 2 runs the model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standard_g2p.word_segmentation import segment, split_words, has_segmenter

SENTENCES = [
    ('th', 'ประเทศต่าง ๆ ใช้ดาวเทียมเพื่อการสื่อสาร'),   # ๆ repeats the previous word
    ('km', 'ខ្ញុំចង់ទៅផ្សារនៅថ្ងៃស្អែក។'),
    ('my', 'မြန်မာနိုင်ငံ၏ အကြီးဆုံးမြို့ ဖြစ်သည်။'),     # ၏ is read aloud, not punctuation
    ('ja', '今日は東京へ行きます。'),                      # は/へ are read wa/e
    ('zh', '音乐会在下午三点开始。'),                      # 乐 is yuè in 音乐, lè in 快乐
    ('yue', '我哋一齊去公園行吓啦。'),
]

# 1. Segmentation, no model.
for lang, text in SENTENCES:
    print('%-4s split_words %s' % (lang, split_words(text)))
    print('     segment     %s' % segment(text, lang))
# Spaced languages go through split_words unchanged.
assert segment('uma parceria', 'pt-BR') == split_words('uma parceria')
# Min Nan and Isan have no segmenter; g2p_task skips them.
print('has_segmenter  nan=%s  tts=%s' % (has_segmenter('nan'), has_segmenter('tts')))

# 2. Through the model: one IPA string per word. For the tonal languages every
#    word carries a tone (an index into TONE_VOCAB).
import torch

from standard_g2p.gold_g2p import goldG2P, TONE_VOCAB, is_tonal

G = goldG2P(device='cuda:0' if torch.cuda.is_available() else 'cpu')
for lang, text in SENTENCES:
    d = G.phonemize_sentence(text, lang=lang)
    print('\n' + text)
    for i, (w, ipa) in enumerate(zip(d['words'], d['ipa'])):
        tones = [TONE_VOCAB[t] for t, n in zip(d['tone'], d['word_num']) if n == i and t]
        print('   %-12s %-26s tones %s' % (w, ipa, tones))
        if is_tonal(lang, diacritic_tone=False):
            assert tones, 'every %s word should carry a tone' % lang

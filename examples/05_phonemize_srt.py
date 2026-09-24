"""File level: a Whisper-style transcript JSON in, a .gs.json of phoneme layers out.

    python examples/05_phonemize_srt.py

For a whole corpus, see `task_g2p_phonemize` in g2p_task.py: it groups clips by
language and reuses one model instance, whose word cache persists across files.
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from standard_g2p.gold_g2p import goldG2P

srt = {
    'lang': 'en-US',
    'audio_path': 'clip.wav',
    'segments': [{
        'start': 0.0, 'end': 1.4, 'text': 'hello world',
        'words': [{'word': 'hello', 'start': 0.0, 'end': 0.6, 'probability': 0.9},
                  {'word': 'world', 'start': 0.7, 'end': 1.4, 'probability': 0.9}],
        # Optional [leading, trailing] silence in seconds. Above the thresholds
        # a SIL token is added at that edge.
        'trim': [0.4, 0.1],
    }],
}

tmp = tempfile.mkdtemp()
srt_path = os.path.join(tmp, 'clip.srt.json')
gs_path = os.path.join(tmp, 'clip.gs.json')
with open(srt_path, 'w', encoding='utf-8') as f:
    json.dump(srt, f)

G = goldG2P(device='cuda:0' if torch.cuda.is_available() else 'cpu')
# `lang` omitted: the file's own 'lang' field is used.
G.phonemize_srt(srt_path, gs_path)

with open(gs_path, encoding='utf-8') as f:
    gs = json.load(f)
print(json.dumps(gs, ensure_ascii=False, indent=1))

# The header carries the scope of every indexed layer in the segments.
seg = gs['segments'][0]
assert all(0 <= i < gs['n_gold_ph'] for i in seg['gold_ph'])
assert len(seg['phonemes']) == len(seg['tone']) == len(seg['stress']) \
    == len(seg['length']) == len(seg['word_num'])

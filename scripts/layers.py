"""Full enumeration of every value each layer can take, no reduction.

Step 1 of rebuilding the gold inventory. Reads the CharsiuG2P training split
(data/train/*.tsv) and writes tmp/inventory_build/layers.json, which
scripts/build_inventory.py consumes.

    python scripts/layers.py
"""
import sys, os, glob, json, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'standard_g2p'))
from gold_g2p import decompose_ipa

D = os.path.join(ROOT, 'tmp', 'inventory_build')
os.makedirs(D, exist_ok=True)

FOCUS = ['eng-us', 'eng-uk', 'zho-s', 'zho-t', 'yue', 'nan', 'tha', 'vie-n', 'vie-s',
         'ara', 'egy', 'hin', 'urd', 'jpn', 'kor', 'spa', 'fra', 'ger', 'rus',
         'por-bz', 'ita', 'tur', 'ind', 'fas', 'bur', 'swa', 'pol', 'dut']

seg = collections.Counter()
tone = collections.Counter()
stress = collections.Counter()
length = collections.Counter()
per_lang = {}
seg_langs = collections.defaultdict(set)
tone_langs = collections.defaultdict(set)

files = sorted(glob.glob(os.path.join(ROOT, 'data/train', '*.tsv')))
for f in files:
    lang = os.path.basename(f)[:-4]
    ls, lt, lst, ll = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) < 2:
                continue
            for var in p[1].split(','):
                var = var.strip()
                if not var:
                    continue
                d = decompose_ipa(var, lang=lang)
                ls.update(d['segments'])
                ll.update(d['length'])
                lst.update(d['stress'])
                lt.update(t for t in d['tone'] if t)
    seg.update(ls); tone.update(lt); stress.update(lst); length.update(ll)
    for s in ls: seg_langs[s].add(lang)
    for t in lt: tone_langs[t].add(lang)
    tot = sum(ls.values())
    per_lang[lang] = {
        'seg_all': len(ls),
        'seg_solid': sum(1 for n in ls.values() if n >= max(5, tot * 1e-4)),
        'tone': len(lt),
        'has_stress': sum(v for k, v in lst.items() if k) > 0,
        'n': tot,
    }
    print('%-10s seg=%4d solid=%4d tone=%3d' % (lang, len(ls), per_lang[lang]['seg_solid'], len(lt)), flush=True)

print('\n' + '=' * 70)
print('LAYER 1 - SEGMENTS (length/tone/stress removed)')
print('=' * 70)
tot = sum(seg.values())
print('unique segments        : %d' % len(seg))
for thr, lbl in [(1, 'all'), (10, '>=10 occ'), (100, '>=100 occ')]:
    print('  %-12s : %d' % (lbl, sum(1 for v in seg.values() if v >= thr)))
print('  in >=2 languages   : %d' % sum(1 for s in seg if len(seg_langs[s]) >= 2))
print('  in >=5 languages   : %d' % sum(1 for s in seg if len(seg_langs[s]) >= 5))
items = seg.most_common()
for r in [50, 100, 150, 200, 250, 300, 400, 500]:
    if r <= len(items):
        print('  top %4d -> %.4f%% of tokens' % (r, 100 * sum(n for _, n in items[:r]) / tot))

print('\n--- FULL SEGMENT LIST (by frequency) ---')
print(' '.join(p for p, _ in items))

print('\n' + '=' * 70)
print('LAYER 2 - TONE')
print('=' * 70)
print('unique tone contours   : %d' % len(tone))
print('tonal languages        : %d' % sum(1 for v in per_lang.values() if v['tone']))
print('\n--- FULL TONE LIST (contour, occurrences, languages) ---')
for t, n in tone.most_common():
    print('  %-10s %9d   %s' % (t, n, ','.join(sorted(tone_langs[t]))[:60]))

print('\n' + '=' * 70)
print('LAYER 3 - STRESS / LAYER 4 - LENGTH')
print('=' * 70)
print('stress values:', dict(stress))
print('length values:', dict(length))
print('languages using stress: %d / %d' % (sum(1 for v in per_lang.values() if v['has_stress']), len(per_lang)))

print('\n' + '=' * 70)
print('FOCUS LANGUAGES')
print('=' * 70)
print('%-9s %8s %8s %6s %8s' % ('lang', 'segments', 'solid', 'tones', 'stress'))
for lg in FOCUS:
    if lg in per_lang:
        v = per_lang[lg]
        print('%-9s %8d %8d %6d %8s' % (lg, v['seg_all'], v['seg_solid'], v['tone'], 'yes' if v['has_stress'] else '-'))

print('\nsizes across all 100 langs: mean seg %.1f, max %d'
      % (sum(v['seg_all'] for v in per_lang.values()) / len(per_lang),
         max(v['seg_all'] for v in per_lang.values())))

json.dump({'segments': seg.most_common(), 'tone': tone.most_common(),
           'stress': dict(stress), 'length': dict(length),
           'per_lang': per_lang,
           'seg_langs': {k: sorted(v) for k, v in seg_langs.items()}},
          open(os.path.join(D, 'layers.json'), 'w'), ensure_ascii=False)
print('\nwrote layers.json')

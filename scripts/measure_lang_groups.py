"""Measure the language statistics behind the groupings in standard_g2p/lang_codes.py.

Everything asserted in lang_codes.py about tone, word segmentation and language
grouping comes from here, so that none of it rests on a typology reference that
may or may not describe the corpus the model was actually trained on.

Reads the CharsiuG2P training dictionaries and writes two files:

    standard_g2p/mappings/lang_stats.json   machine readable, for regenerating tables
    standard_g2p/mappings/lang_stats.md     the same numbers written up

Sections 1-2 are per CharsiuG2P tag, because what they measure is each tag's
own dictionary; every row also carries the BCP 47 code the rest of the pipeline
uses for that tag. Sections 3-5 are per language and name languages by BCP 47
code only, counting just the tags a BCP 47 code reaches ('isl' and 'slo'
duplicate 'ice' and 'slk' and are never emitted, so counting them would
double-weight Icelandic and Slovak).

Phoneme inventories are measured in the gold inventory's space
(phoneme_inventory_gold, via the same decompose_ipa -> map_phoneme path the
G2P uses), because that is the space every model is trained in. Raw IPA would
count 'd̥' and 'd' as different phonemes where the models cannot.

Usage:
    python scripts/measure_lang_groups.py [--dicts DIR] [--sample N]

The dictionaries are one `word<TAB>ipa` file per CharsiuG2P tag, in this
repo's `dicts/` (inherited from https://github.com/lingjzhu/CharsiuG2P).
"""
import os
import sys
import json
import random
import argparse
import collections
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'standard_g2p'))

from gold_g2p import decompose_ipa                        # noqa: E402
import phoneme_inventory_gold as PI                         # noqa: E402
from lang_codes import (TAG_TO_ISO, ISO_TO_TAG, ISO_TO_GROUP,  # noqa: E402
                        FAMILY_ISO, tag_to_bcp47, bcp47_to_tag)

DEFAULT_DICTS = os.path.join(ROOT, 'dicts')

# Tone is written three different ways across the corpus; all three are counted
# separately because they do not reach the `tone` layer equally (see
# lang_codes.TONE_LETTER_ISO vs TONE_DIACRITIC_ISO).
TONE_LETTERS = set('˥˦˧˨˩')            # ˥˦˧˨˩
CHAO_DIGITS = set('⁰¹²³⁴⁵')       # ⁰¹²³⁴⁵ (nan)
TONE_DIACRITICS = set('́̀̂̌̄')         # ́  ̀  ̂  ̌  ̄

SCRIPTS = ('LATIN', 'CYRILLIC', 'GREEK', 'ARABIC', 'HEBREW', 'DEVANAGARI', 'TAMIL',
           'ORIYA', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'TELUGU', 'KANNADA',
           'MALAYALAM', 'SINHALA', 'THAI', 'LAO', 'KHMER', 'MYANMAR', 'CJK',
           'HIRAGANA', 'KATAKANA', 'HANGUL', 'ARMENIAN', 'GEORGIAN', 'ETHIOPIC',
           'SYRIAC', 'TIBETAN', 'THAANA')


def script_of(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for s in SCRIPTS:
        if name.startswith(s) or (' ' + s) in name:
            return s
    return None


def read_dict(path, sample, rng):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and parts[0] and parts[1]:
                rows.append((parts[0], parts[1]))
    total = len(rows)
    if sample and total > sample:
        rows = rng.sample(rows, sample)
    return rows, total


def measure_one(tag, rows, total):
    """Per-language counts: tone marking, key length, script, phoneme inventory."""
    tone_letter = chao = diacritic = 0
    keylen = collections.Counter()
    scripts = collections.Counter()
    inv = collections.Counter()

    for word, ipa in rows:
        keylen[min(len(word), 3)] += 1          # bucket: 1, 2, 3+
        for ch in word:
            s = script_of(ch)
            if s:
                scripts[s] += 1

        if any(c in TONE_LETTERS for c in ipa):
            tone_letter += 1
        elif any(c in CHAO_DIGITS for c in ipa):
            chao += 1
        if any(c in TONE_DIACRITICS for c in unicodedata.normalize('NFD', ipa)):
            diacritic += 1

        # Gold units, exactly as goldG2P._gold_map produces them; stress,
        # tone and length are separate layers and never reach the inventory.
        try:
            for seg in decompose_ipa(ipa.split(',')[0], lang=tag)['segments']:
                inv.update(u for u in PI.map_phoneme(seg)
                           if u not in PI.SPECIAL_TOKENS)
        except Exception:
            pass

    n = max(len(rows), 1)
    main_script = scripts.most_common(1)[0][0] if scripts else None
    if main_script in ('HIRAGANA', 'KATAKANA', 'CJK') and tag == 'jpn':
        main_script = 'JPN-MIXED'

    return {
        'tag': tag,
        'bcp47': tag_to_bcp47(tag),
        'reachable': bcp47_to_tag(tag_to_bcp47(tag)) == tag,
        'iso': TAG_TO_ISO[tag],
        'entries_total': total,
        'entries_sampled': len(rows),
        'pct_tone_letter': 100.0 * tone_letter / n,
        'pct_chao_digit': 100.0 * chao / n,
        'pct_tone_diacritic': 100.0 * diacritic / n,
        'pct_key_1char': 100.0 * keylen[1] / n,
        'pct_key_2char': 100.0 * keylen[2] / n,
        'pct_key_3plus': 100.0 * keylen[3] / n,
        'main_script': main_script,
        'script_counts': dict(scripts.most_common(4)),
        'inventory_size': len(inv),
        'inventory': dict(inv),
    }


def cluster(per_iso_freq, isos):
    """Do phoneme inventories recover language families? Two metrics, both no."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    vocab = sorted({p for c in per_iso_freq.values() for p in c})
    idx = {p: i for i, p in enumerate(vocab)}
    X = np.zeros((len(isos), len(vocab)))
    for r, iso in enumerate(isos):
        c = per_iso_freq[iso]
        tot = sum(c.values()) or 1
        for p, v in c.items():
            X[r, idx[p]] = v / tot

    out = {'n_languages': len(isos), 'n_phoneme_types': len(vocab)}

    # (a) Jaccard over which phonemes are used at all, above a 0.1% floor.
    B = (X > 0.001).astype(float)
    inter = B @ B.T
    size = B.sum(1)
    jac = inter / (size[:, None] + size[None, :] - inter)
    Dj = 1 - jac
    Zj = linkage(squareform(Dj, checks=False), method='average')
    out['jaccard'] = {
        'mean_pairwise_similarity': float((jac.sum() - len(isos)) / (len(isos) * (len(isos) - 1))),
        'cluster_sizes': {str(k): sorted(collections.Counter(
            fcluster(Zj, k, criterion='maxclust')).values(), reverse=True)
            for k in (6, 7, 8, 9, 10)},
    }

    # (b) Jensen-Shannon over how often each phoneme is used.
    def jsd(P, Q):
        M = 0.5 * (P + Q)
        def kl(a, b):
            m = a > 0
            return float(np.sum(a[m] * np.log2(a[m] / b[m])))
        return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

    n = len(isos)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = jsd(X[i], X[j])
    Z = linkage(squareform(D, checks=False), method='ward')
    memb = {}
    for k in (6, 7, 8, 9, 10):
        lab = fcluster(Z, k, criterion='maxclust')
        g = collections.defaultdict(list)
        for iso, l in zip(isos, lab):
            g[int(l)].append(iso)
        memb[str(k)] = {str(l): sorted(v) for l, v in sorted(g.items())}
    out['jensen_shannon'] = {
        'cluster_sizes': {k: sorted((len(v) for v in m.values()), reverse=True)
                          for k, m in memb.items()},
        'memberships': memb,
    }

    # Nearest neighbour per language, which is what a family-based grouping
    # would have to reproduce to be justified.
    nn = {}
    for i, iso in enumerate(isos):
        order = np.argsort(D[i])
        j = int(order[1])
        nn[iso] = {'nearest': isos[j], 'jsd': float(D[i, j])}
    out['nearest_neighbour'] = nn
    return out


def agreement(clustering, iso_of):
    """How far the clusters line up with genetic family and processing group.

    Computed rather than asserted in the prose, so the write-up cannot drift
    from the numbers when the measurement changes. `iso_of` maps each label
    back to the ISO 639-3 key FAMILY_ISO / ISO_TO_GROUP use.
    """
    top = lambda code: FAMILY_ISO[iso_of[code]].split(':')[0]     # noqa: E731
    fine = lambda code: FAMILY_ISO[iso_of[code]]                  # noqa: E731
    grp = lambda code: ISO_TO_GROUP[iso_of[code]]                 # noqa: E731

    nn = clustering['nearest_neighbour']
    n = len(nn)
    out = {'n': n,
           'nn_same_family': sum(top(a) == top(v['nearest']) for a, v in nn.items()),
           'nn_same_subfamily': sum(fine(a) == fine(v['nearest']) for a, v in nn.items()),
           'nn_same_group': sum(grp(a) == grp(v['nearest']) for a, v in nn.items())}

    # Per JSD cluster at k=8: its majority family and how much of it that is.
    purity = {}
    for label, members in clustering['jensen_shannon']['memberships']['8'].items():
        fam, k = collections.Counter(top(m) for m in members).most_common(1)[0]
        purity[label] = {'family': fam, 'count': k, 'size': len(members)}
    out['jsd_k8_purity'] = purity
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dicts', default=DEFAULT_DICTS)
    ap.add_argument('--sample', type=int, default=12000,
                    help='max entries per language (0 = all)')
    ap.add_argument('--out-json', default=os.path.join(ROOT, 'standard_g2p', 'mappings', 'lang_stats.json'))
    ap.add_argument('--out-md', default=os.path.join(ROOT, 'standard_g2p', 'mappings', 'lang_stats.md'))
    args = ap.parse_args()

    rng = random.Random(0)
    per_tag = {}
    missing = []
    for tag in sorted(TAG_TO_ISO):
        path = os.path.join(args.dicts, tag + '.tsv')
        if not os.path.exists(path):
            missing.append(tag)
            continue
        rows, total = read_dict(path, args.sample, rng)
        per_tag[tag] = measure_one(tag, rows, total)
        print(f'  {tag:10s} {total:7d} entries  {per_tag[tag]["inventory_size"]:3d} phonemes',
              file=sys.stderr)

    # Variant tags of one language share a row (en + en-GB -> en), labelled by
    # the language's bare BCP 47 code. Unreachable duplicate tags are left out.
    per_iso = collections.defaultdict(collections.Counter)
    for t, v in per_tag.items():
        if v['reachable']:
            per_iso[v['iso']].update(v['inventory'])
    label = {iso: tag_to_bcp47(ISO_TO_TAG[iso]) for iso in per_iso}
    iso_of = {code: iso for iso, code in label.items()}
    per_lang = {label[iso]: c for iso, c in per_iso.items()}
    langs = sorted(per_lang)
    clustering = cluster(per_lang, langs)

    stats = {
        'source': args.dicts,
        'sample_per_language': args.sample,
        'inventory_space': f'phoneme_inventory_gold ({PI.N_TOKENS} tokens)',
        'missing_dicts': missing,
        'n_tags': len(per_tag),
        'unreachable_tags': sorted(t for t, d in per_tag.items() if not d['reachable']),
        'n_languages': len(langs),
        'per_tag': {t: {k: v for k, v in d.items() if k != 'inventory'}
                    for t, d in per_tag.items()},
        'clustering': clustering,
        'agreement': agreement(clustering, iso_of),
    }

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    write_markdown(stats, per_tag, args.out_md)
    print(f'wrote {args.out_json} and {args.out_md}', file=sys.stderr)


def write_markdown(stats, per_tag, path):
    L = []
    A = L.append
    A('# Language statistics behind the groupings')
    A('')
    A('Generated by `scripts/measure_lang_groups.py` -- do not edit by hand.')
    A('')
    A(f'Source: CharsiuG2P training dictionaries (`{stats["source"]}`), '
      f'{stats["n_tags"]} tags / {stats["n_languages"]} languages, '
      f'sampled at most {stats["sample_per_language"]} entries per tag '
      f'(seed 0). Languages are named by BCP 47 code. Tags no BCP 47 code '
      f'reaches ({", ".join(f"`{t}`" for t in stats["unreachable_tags"])}) '
      f'appear in sections 1-2 only.')
    A('')

    A('## 1. Tone marking')
    A('')
    A('How tone is written in the corpus, which is what decides whether it reaches')
    A('the `tone` layer. Only rows above 1% are listed.')
    A('')
    A('| tag | BCP 47 | ISO | entries | tone letters | Chao digits | diacritics |')
    A('|---|---|---|---:|---:|---:|---:|')
    for t, d in sorted(per_tag.items()):
        if max(d['pct_tone_letter'], d['pct_chao_digit'], d['pct_tone_diacritic']) < 1.0:
            continue
        A(f'| `{t}` | `{d["bcp47"]}` | `{d["iso"]}` | {d["entries_total"]:,} | '
          f'{d["pct_tone_letter"]:.1f}% | {d["pct_chao_digit"]:.1f}% | '
          f'{d["pct_tone_diacritic"]:.1f}% |')
    A('')
    A('Tone letters and Chao digits go to the `tone` layer. Diacritics do not, by')
    A('default -- and for `hbs`/`slv`/`san`/`grc` they mark pitch accent and for')
    A('`kur` stress, not tone, which is why a diacritic alone does not qualify a')
    A('language as tonal.')
    A('')

    A('## 2. Dictionary key length')
    A('')
    A('The unit the dictionary is keyed by, which decides how much word')
    A('segmentation matters: a language whose entries are nearly all multi-character')
    A('cannot be looked up character by character.')
    A('')
    A('| tag | BCP 47 | ISO | entries | 1 char | 2 chars | 3+ chars |')
    A('|---|---|---|---:|---:|---:|---:|')
    for t in ('zho-s', 'zho-t', 'yue', 'nan', 'jpn', 'tha', 'khm', 'bur', 'tts'):
        if t not in per_tag:
            continue
        d = per_tag[t]
        A(f'| `{t}` | `{d["bcp47"]}` | `{d["iso"]}` | {d["entries_total"]:,} | '
          f'{d["pct_key_1char"]:.1f}% | {d["pct_key_2char"]:.1f}% | '
          f'{d["pct_key_3plus"]:.1f}% |')
    A('')

    A('## 3. Writing system')
    A('')
    byscript = collections.defaultdict(set)
    for d in per_tag.values():
        if d['reachable']:
            byscript[d['main_script']].add(d['bcp47'])
    A('| script | n | languages |')
    A('|---|---:|---|')
    for s in sorted(byscript, key=lambda x: (-len(byscript[x]), str(x))):
        A(f'| {s} | {len(byscript[s])} | {" ".join(sorted(byscript[s]))} |')
    A('')

    A('## 4. Do phoneme inventories cluster by language family?')
    A('')
    A('No. This is the measurement that decided against grouping by genetic family.')
    A('')
    c = stats['clustering']
    ag = stats['agreement']
    A(f'{c["n_languages"]} languages, {c["n_phoneme_types"]} distinct phonemes of the '
      f'{stats["inventory_space"]} -- the space the models are trained in, so '
      f'sub-phonemic detail the inventory backs off (`d̥` -> `d`) does not count '
      f'as a difference. Stress, tone and length are separate layers.')
    A('')
    A('**(a) Jaccard over which phonemes each language uses** (above a 0.1% floor). '
      f'Mean pairwise similarity {c["jaccard"]["mean_pairwise_similarity"]:.2f}; '
      'average-linkage cluster sizes:')
    A('')
    A('| k | cluster sizes |')
    A('|---|---|')
    for k, s in c['jaccard']['cluster_sizes'].items():
        A(f'| {k} | {s} |')
    A('')
    big = max(c['jaccard']['cluster_sizes']['8'])
    A(f'At k=8 the largest cluster holds {big} of {c["n_languages"]} languages: '
      f'the shared IPA core dominates which phonemes a language uses at all.')
    A('')
    A('**(b) Jensen-Shannon over how often each phoneme is used**, Ward linkage. '
      'Better balanced, still not families:')
    A('')
    A('| k | cluster sizes |')
    A('|---|---|')
    for k, s in c['jensen_shannon']['cluster_sizes'].items():
        A(f'| {k} | {s} |')
    A('')
    A('Membership at k=8:')
    A('')
    for l, v in sorted(c['jensen_shannon']['memberships']['8'].items(),
                       key=lambda kv: (-len(kv[1]), kv[1])):
        p = ag['jsd_k8_purity'][l]
        A(f'- **{len(v)}** (majority {p["family"]} {p["count"]}/{p["size"]}): '
          f'{" ".join(v)}')
    A('')
    n_pure = sum(p['count'] == p['size'] for p in ag['jsd_k8_purity'].values())
    A(f'A language\'s nearest neighbour by phoneme usage shares its top-level '
      f'family for {ag["nn_same_family"]}/{ag["n"]} languages, its sub-family for '
      f'{ag["nn_same_subfamily"]}/{ag["n"]}, and its processing group for '
      f'{ag["nn_same_group"]}/{ag["n"]}. {n_pure} of the '
      f'{len(ag["jsd_k8_purity"])} clusters above are a single family; a '
      f'family-based grouping would need them all to be.')
    A('')
    A('**Nearest neighbour by phoneme usage**, per language:')
    A('')
    A('| language | nearest | JSD |')
    A('|---|---|---:|')
    for code, v in sorted(c['nearest_neighbour'].items()):
        A(f'| `{code}` | `{v["nearest"]}` | {v["jsd"]:.3f} |')
    A('')

    A('## 5. The resulting groups')
    A('')
    A('What sections 1-4 were used to decide, as it now stands in')
    A('`standard_g2p/lang_codes.py`. Grouping is by preprocessing path -- writing')
    A('system, word segmentation, tone -- because section 4 ruled out phonology.')
    A('')
    A('Every BCP 47 code the pipeline can emit, which is what')
    A('`mappings/group_inventories.json` routes on; regional and script variants')
    A('of one language (`en`, `en-GB`) are listed separately.')
    A('')
    A('| group | codes | usable | languages |')
    A('|---|---:|---:|---|')
    from lang_codes import PROCESSING_GROUPS, EXCLUDED_ISO
    for g in PROCESSING_GROUPS:
        m = sorted(d['bcp47'] for d in per_tag.values()
                   if d['reachable'] and ISO_TO_GROUP[d['iso']] == g)
        usable = [d['bcp47'] for d in per_tag.values() if d['reachable']
                  and ISO_TO_GROUP[d['iso']] == g and d['iso'] not in EXCLUDED_ISO]
        A(f'| `{g}` | {len(m)} | {len(usable)} | {" ".join(m)} |')
    A('')
    A('Excluded for now, orthogonally to the grouping:')
    A('')
    for iso, why in sorted(EXCLUDED_ISO.items()):
        A(f'- `{tag_to_bcp47(ISO_TO_TAG[iso])}`: {why}')
    A('')
    A('Genetic family is recorded separately in `FAMILY_ISO`, as metadata for')
    A('corpus balance rather than as a processing axis. Distribution:')
    A('')
    fam = collections.Counter(FAMILY_ISO.values())
    A('| family | n |')
    A('|---|---:|')
    for f_, n in fam.most_common():
        A(f'| {f_} | {n} |')
    A('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()

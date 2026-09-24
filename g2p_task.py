"""G2P task: turn transcript SRTs into phoneme-layer JSON.

"""

import os
import sys
import json
import traceback

from tqdm import tqdm


def _resolve(path, metadata_dir):
    """paths_list stores paths as absolute, or '$/...' relative to METADATA_DIR."""
    if not path:
        return path
    if path[:2] == '$/':
        return os.path.join(metadata_dir, path[2:])
    if path[0] != '/' and metadata_dir:
        return os.path.join(metadata_dir, path)
    return path


def task_g2p_phonemize(paths_list,  device='cpu', redo=False,
                       metadata_dir='', batch_size=128, output_ext='.gs.json',
                       skip_unsupported=True):
    """Phonemize every transcribed clip in `paths_list`.

    Each clip's 'lang' is a single BCP 47 code (ISO 639-1 where a language has
    one, else ISO 639-3, e.g. 'en', 'pt-BR', 'zh-Hant') -- see
    standard_g2p.lang_codes.BCP47_TO_TAG for the vocabulary. CharsiuG2P's own tag
    spelling never appears in paths_list; it is resolved internally and only
    shows up as the 'g2p_lang' field of the .gs.json this writes.

    skip_unsupported  skip languages that need word segmentation but have no
                      segmenter (nan, tts) rather than emitting garbage for
                      them. th/km/my/ja/zh/yue are segmented (see
                      standard_g2p/word_segmentation.py) and are not skipped.
    """
    from standard_g2p.gold_g2p import (goldG2P, resolve_tag, N_TONES,
                                       has_segmenter, require_segmenter,
                                       UnsupportedLanguageError)

    G = goldG2P(device=device, batch_size=batch_size)
    print('G2P: inventory of %d tokens, %d tone classes, device %s'
          % (G._inv.N_TOKENS, N_TONES, G.device))

    done = skipped = failed = 0

    unique_langs = {clip['lang'] for clip in paths_list}

    supported_langs = []
    for lang in unique_langs:
        try:
            g2p_tag = resolve_tag(lang, default=G.default_lang)
        except UnsupportedLanguageError as ex:
            print('G2P: unsupported language %s -- %s' % (lang, str(ex)))
            continue
        if skip_unsupported and not has_segmenter(g2p_tag):
            print('G2P: skipping unsupported language %s (tag %s) '
                    'that needs word segmentation' % (lang, g2p_tag))
            continue
        # Import the segmenter package now: if it is missing, the run stops
        # here with the package name, not once per clip in the `failed` count.
        if has_segmenter(g2p_tag):
            require_segmenter(g2p_tag)
        supported_langs.append(lang)
    unique_langs = supported_langs

    print('G2P: unique languages:', unique_langs)

    for lang in unique_langs:
        G.set_default_lang(lang)
        this_lang_paths_list = [p for p in paths_list if p.get('lang') == lang]

        for i in tqdm(range(len(this_lang_paths_list)), leave=False, desc='G2P phonemizing %s' % lang):
            clip = this_lang_paths_list[i]

            srt_path = _resolve(clip.get('srt'), metadata_dir)
            if not srt_path:
                print("Error: clip %d has no srt path" % i, clip.get('srt'))
                failed += 1
                continue
            gs_out_path = srt_path.replace('.srt.json', output_ext).replace('.srt', output_ext)


            if os.path.exists(gs_out_path) and not redo:
                skipped += 1
                continue
            if not os.path.exists(srt_path):
                print('Error: srt not found', srt_path)
                failed += 1
                continue

            try:
                G.phonemize_srt(srt_path, gs_out_path, lang=lang)
                done += 1
            except UnsupportedLanguageError:
                # A bad lang is a data or config bug, not a per-clip fluke --
                # let it stop the run instead of burying it in the `failed`
                # count, where it would silently masquerade as noise.
                raise
            except Exception:
                failed += 1
                print('Error phonemizing', srt_path)
                traceback.print_exc()

    print('Done. written=%d skipped=%d failed=%d from total clips %d' % (done, skipped, failed, len(paths_list)))
    G.print_stats()
    return G


def check_sample(device='cuda:0'):
    """Round-trip a synthetic SRT so the format can be eyeballed."""
    import tempfile
    from standard_g2p.gold_g2p import goldG2P

    srt = {
        'lang': 'en-US',
        'audio_path': 'tmp/data/audio_samples/test1.wav',
        'segments': [{
            'start': 0.0, 'end': 1.4, 'text': 'hello world',
            'words': [{'word': 'hello', 'start': 0.0, 'end': 0.6, 'probability': 0.9},
                      {'word': 'world', 'start': 0.7, 'end': 1.4, 'probability': 0.9}],
        }],
    }
    d = tempfile.mkdtemp()
    srt_path, gs_out_path = os.path.join(d, 'x.srt'), os.path.join(d, 'x.ts')
    with open(srt_path, 'w') as f:
        json.dump(srt, f)

    G = goldG2P(device=device)
    G.phonemize_srt(srt_path, gs_out_path, lang='en-US')
    with open(gs_out_path) as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=1))
    G.print_stats()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    check_sample('cpu')

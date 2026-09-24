
import json

from .g2p_task import task_g2p_phonemize



device = 'cpu'

METADATA_DIR = "/mnt/intelpa-3/rehman/common_dir"


def main():
    paths_list_dev = "/mnt/intelpa-2/rehman/codes/speech_data_manager/paths_cache/paths_list_fleurs_dev_1h.json"

    with open(paths_list_dev, "r") as f:
        paths_list = json.load(f)


    task_g2p_phonemize(paths_list,
                        device=device, redo=True,
                        metadata_dir=METADATA_DIR,
                        batch_size=128, output_ext='.gs.json',
                        skip_unsupported=True)
    exit()

if __name__ == '__main__':
    main()

from pathlib import Path
import json
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
root = Path(current).parent.parent
sys.path.append(str(root))
from utils import set_seed


def setting(dataset_name: str = None, project: str = 'GuardianNet'):

    BASE_DIR = Path(__file__).resolve().parent.parent
    Path(os.path.join(root, 'session', f'session_{project}')).mkdir(exist_ok=True)
    Path(os.path.join(BASE_DIR, 'trained_ae')).mkdir(exist_ok=True)

    i = 1
    flag = True
    project = 'GuardianNet'
    SAVE_PATH_ = str()
    CHECKPOINT_PATH_ = str()

    config_name = f'CONFIG_{dataset_name}'
    config_dir = os.path.join(BASE_DIR, 'configs')
    config = open(os.path.join(config_dir, f'{config_name}.json'))
    config = json.load(config)

    while flag:

        TEMP_FILENAME = f"{dataset_name}-{config['classifier']['name']}-{i}"
        TEMP_PATH = os.path.join(root, 'session', f'session_{project}', TEMP_FILENAME)

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(os.path.join(TEMP_PATH))
            SAVE_PATH_ = BASE_DIR.joinpath(TEMP_PATH)

            os.mkdir(os.path.join(BASE_DIR, SAVE_PATH_, 'model_checkpoint'))
            CHECKPOINT_PATH_ = os.path.join(BASE_DIR, SAVE_PATH_, 'model_checkpoint')

            with open(f'{SAVE_PATH_}/CONFIG_FILE.json', 'w') as f:
                json.dump(config, f)

            time_file = open(SAVE_PATH_.joinpath('time.txt'), 'w')
            time_file.write('Result Time \n')

            print(f'MODEL SESSION: {SAVE_PATH_}')

    config['save_path'] = SAVE_PATH_
    config['checkpoint_path'] = CHECKPOINT_PATH_

    seed = config['seed']
    if seed is not None:
        set_seed(seed=seed)
    print(f'SEED NUMBER SET TO {seed}')

    return config

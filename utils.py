import itertools
from copy import deepcopy
import torch

EMPTY = lambda x: x

def get_set(config):
    if isinstance(config, list):
        return config
    elif isinstance(config, dict):
        if 'stop' in config:
            return range(config.get('start', 0), config['stop'], config.get('step', 1))
        else:
            return itertools.count(config.get('start', 0), config.get('step', 1))
    else:
        raise ValueError

def set_env(gpuid=0, detect_anomaly=False, deterministic=False) -> torch.device:
    device = torch.device('cuda', index=gpuid) \
        if torch.cuda.is_available() else torch.device('cpu')
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False
    return device



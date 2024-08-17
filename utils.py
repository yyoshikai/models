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

def get_device(gpuid: int) -> torch.device:
    device = torch.device('cuda', index=gpuid) \
        if torch.cuda.is_available() else torch.device('cpu')
    return device
    
def set_env(detect_anomaly=False, deterministic=False) -> None:
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False



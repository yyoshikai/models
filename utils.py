import random
import numpy as np
import torch


def check_leftargs(self, logger, kwargs, show_content=False):
    if len(kwargs) > 0:
        raise ValueError(f"Unknown kwarg in {type(self).__name__}: {list(kwargs.keys())}")

EMPTY = lambda x: x

class GlobalRandomState:
    def __init__(self, seed=None):
        if seed is not None:
            self.set_seed(seed)

    def state_dict(self):
        return {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()
        }
    
    def load_state_dict(self, state_dict):
        random.setstate(state_dict['random'])
        np.random.set_state(state_dict['numpy'])
        torch.set_rng_state(state_dict['torch'])
        torch.cuda.set_rng_state_all(state_dict['cuda'])

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
import sys, os
import pickle
import time
import random
import shutil
from logging import getLogger
from logging.config import dictConfig
from copy import deepcopy
from contextlib import nullcontext
from collections import OrderedDict

import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("TOOLS_DIR", "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
from tools.notice import notice, noticeerror
from tools.path import make_result_dir, timestamp
from tools.tools import Singleton

from tools.args import load_config2, subs_vars
from tools.torch import get_params
from models.dataset import DataLoader
from models.accumulator import get_accumulator, NumpyAccumulator
from models.metric import get_metric
from models.process2 import get_processes
from models import Model
from models.optimizer import ModelOptimizer
from models.utils import set_deterministic, get_device
from models.alarm import Alarm

class RState(Singleton):
    def __init__(self, seed: int=None):
        if seed is not None:
            self.init_seed(seed)
    
    def init_seed(self, seed: int):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
    
    def state_dict(self):
        return OrderedDict(
            random=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()
        )
    
    def load_state_dict(self, state_dict: dict[str]):
        random.setstate(state_dict['random'])
        np.random.set_state(state_dict['numpy'])
        torch.set_rng_state(state_dict['torch'])
        torch.cuda.set_rng_state_all(state_dict['cuda'])

@noticeerror(from_=f"train.py in {os.getcwd()}", notice_end=False)
def main(
        result_dir: dict,
        log: dict,
        model: dict,
        data: dict,
        process: dict,

        version: float=0.0,
        
        gpuid: int=0,
        deterct_anomaly: bool=False,
        deterministic: bool=False,
        init_weight: dict={},
        model_seed=None,
        
        config_: dict={},
        args=None,
    ):

    # Environment    
    assert version >= 1.0
    logger = getLogger(__name__)
    dictConfig(log)
    result_dir = make_result_dir(**result_dir)
    with open(f"{result_dir}/config.yaml", mode='w') as f:
        yaml.dump(config_, f, sort_keys=False)
    logger.warning(f"options: {' '.join(args)}")
    torch.autograd.set_detect_anomaly = deterct_anomaly
    set_deterministic(deterministic)
    device = get_device(gpuid)
    logger.warning(f"device: {device}")
    rstate = RState()

    # Data
    dl = DataLoader(**data)

    # Model
    rstate.init_seed(model_seed)
    model = Model(**model)
    model.load(**init_weight)
    model.to(device)
    ## Show params
    df_param, n_param, bit_size = get_params(model)
    df_param.to_csv(f"{result_dir}/parameters.tsv", sep='\t', index=False)
    logger.info(f"# of params: {n_param}")
    logger.info(f"Model size(bit): {bit_size}")

    ## Process
    processes = get_processes(processes)
    processes['train']
    


if __name__ == '__main__':
    config_ = load_config2("", default_configs=['base.yaml'])
    ## replacement of config: add more when needed
    config_ = subs_vars(config_, {"$TIMESTAMP": timestamp()})
    main(args=sys.argv, config_=config_.to_dict(), **config_)



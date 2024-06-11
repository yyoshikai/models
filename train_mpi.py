# mpiではないが, train.pyをiterationする。
import sys, os
import yaml
import itertools
from copy import deepcopy

from addict import Dict

TOOLS_DIR = os.environ.get('TOOLS_DIR', "/workspace")
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)
from models.train import main
from tools.args import subs_vars
from tools.args import load_config2, load_config3

config = load_config2("", default_configs=[])

base_args = list(map(str, config.base))
iter_keys = list(config.iterations.keys())
iterations = [config.iterations[key] for key in iter_keys]
iter_values = list(itertools.product(*iterations))
for iter_value in iter_values[config.gpuid::config.gpusize]:
    iter_vars = {}
    for key, value in zip(iter_keys, iter_value):
        if ';' in key:
            for key0, value0 in zip(key.split(';'), value):
                iter_vars[key0] = str(value0)
        else:
            iter_vars[key] = str(value)
    args = deepcopy(base_args)
    args = subs_vars(args, iter_vars)
    config0 = load_config3(args, "", default_configs=['base.yaml'])
    main(config0, args)






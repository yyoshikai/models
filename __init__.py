import os
import subprocess
import functools


def get_git_version():
    git_out = subprocess.run('git branch', shell=True, 
        capture_output=True, text=True, cwd=os.path.dirname(__file__))
    if git_out.returncode != 0:
        return None
    for line in git_out.stdout.splitlines():
        if line[0] == '*':
            return line.split(' ')[1]
    else:
        return None

print(f"[NOTICE] Git branch of models is '{get_git_version()}'.")

module_type2class = {}
def register_module(cls):
    if cls.__name__ in module_type2class:
        raise ValueError(f"{cls.__name__} was registered for multiple times.")
    module_type2class[cls.__name__] = cls
    return cls

from .models import *
from .dataset_old import get_dataloader
from .process import get_process, get_processes

import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

for cls in [nn.Linear, nn.ReLU, nn.GELU, nn.LayerNorm, nn.BatchNorm1d]:
    module_type2class[cls.__name__] = cls

from .modules.sequence import *

from .modules.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler, NemotoPooler, 
    GraphPooler, GraphMeanMaxPooler, GraphStartMeanMaxPooler, GraphPooler2]:
    module_type2class[cls.__name__] = cls

from .modules.graph_transformer import *
for cls in [GraphAttentionLayer, GraphEncoder, AtomEmbedding]:
    module_type2class[cls.__name__] = cls

from .modules import grover, image, unimol, unimol2, vae, criterion


from .modules.ssl import *
for cls in [BarlowTwinsCriterion, MolCLIPCriterion]:
    module_type2class[cls.__name__] = cls


# dataset
from .dataset_old import dataset_type2class
from .data.grover import GroverDataset, Grover2Dataset
dataset_type2class['grover'] = GroverDataset
dataset_type2class['grover2'] = Grover2Dataset

# function
from .models import function_name2func

from .modules.sequence import get_token_size
for func in [get_token_size]:
    function_name2func[func.__name__] = func

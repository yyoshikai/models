import functools

module_type2class = {}
def register_module(cls):
    if cls.__name__ in module_type2class:
        raise ValueError(f"{cls.__name__} was registered for multiple times.")
    module_type2class[cls.__name__] = cls
    return cls

from .models2 import *
from .dataset import get_dataloader
from .process import get_process

import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.sequence import *

from .modules2.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler, NemotoPooler, 
    GraphPooler, GraphMeanMaxPooler, GraphStartMeanMaxPooler, GraphPooler2]:
    module_type2class[cls.__name__] = cls

from .modules2.graph_transformer import *
for cls in [GraphAttentionLayer, GraphEncoder, AtomEmbedding]:
    module_type2class[cls.__name__] = cls

from .modules2 import grover, image, unimol, unimol2, vae, criterion


from .modules2.ssl import *
for cls in [BarlowTwinsCriterion, MolCLIPCriterion]:
    module_type2class[cls.__name__] = cls


# dataset
from .dataset import dataset_type2class
from .datasets.grover import GroverDataset, Grover2Dataset
dataset_type2class['grover'] = GroverDataset
dataset_type2class['grover2'] = Grover2Dataset

# function
from .models2 import function_name2func

from .modules2.sequence import get_token_size
for func in [get_token_size]:
    function_name2func[func.__name__] = func

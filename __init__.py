from  .models2 import *

import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.tunnel import *
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

from .modules2.sequence import *
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, AttentionDecoder, TransformerLMDecoder,
    GreedyDecoder, CrossEntropyLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.vae import *
for cls in [VAE, MinusD_KLLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler, NemotoPooler, 
    GraphPooler, GraphMeanMaxPooler, GraphStartMeanMaxPooler, GraphPooler2]:
    module_type2class[cls.__name__] = cls

from .modules2.graph_transformer import *
for cls in [GraphAttentionLayer, GraphEncoder, AtomEmbedding]:
    module_type2class[cls.__name__] = cls

from .modules2.unimol import *
for cls in [UnimolEncoder, UnimolEncoder2, UnimolEmbedding, UnimolGraphEmbedding,
    UnimolETEmbedding, DummyAdder]:
    module_type2class[cls.__name__] = cls

from .modules2.ssl import *
for cls in [BarlowTwinsCriterion, MolCLIPCriterion]:
    module_type2class[cls.__name__] = cls

from .modules2.grover import *
for cls in [GroverEncoder, GTransEncoder, Grover2UnimolEmbedding]:
    module_type2class[cls.__name__] = cls

# dataset
from .dataset import dataset_type2class
from .datasets.grover import GroverDataset, Grover2Dataset
dataset_type2class['grover'] = GroverDataset
dataset_type2class['grover2'] = Grover2Dataset
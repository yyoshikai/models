from  .models2 import *

from .tunnel import *
for cls in [Tunnel, Function]:
    module_type2class[cls.__name__] = cls

from .modules2.sequence import *
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, GreedyDecoder, CrossEntropyLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler, NemotoPooler]:
    module_type2class[cls.__name__] = cls
from .modules2.vae import *
for cls in [VAE]:
    module_type2class[cls.__name__] = cls

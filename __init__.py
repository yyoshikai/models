from  .models2 import *

from .modules2.sequence import *
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, Tunnel, CrossEntropyLoss]:
    module_type2class[cls.__name__] = cls


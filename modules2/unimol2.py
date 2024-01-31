"""
240119作成
Unimolのdecoder
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sequence import get_posenc


class UnimolDecoder(nn.Module):
    def __init__(self, 
                layer: dict,
                n_layer: int, 
                max_len: int):
        super().__init__()
        d_model = layer['embed_dim']
        self.layers = nn.ModuleList([UnimolDecoderLayer(**layer) for _ in range(n_layer)])

        self.pe = get_posenc(max_len, d_model)

    def forward(self, memory, memory_key_padding_mask, lengths):
        """
        Parameters
        ----------
        memory(float)[length, batch_size, d_model]
        
        memory_key_padding_mask(float)[batch_size, length]
            sequenceのものと同じ。

        lengths: (long)[batch_size]

        """
        mlength, batch_size, _ = memory.shape

        atoms_emb = self.pe[:mlength]
        



import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models2 import init_config2func
import matplotlib.pyplot as plt

class GraphAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, edge_voc_size, edge_pad_token, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.edge_embedding = nn.Embedding(edge_voc_size, num_heads, padding_dx=edge_pad_token)

    def forward(self, x: torch.Tensor, edge: torch.Tensor,
                key_padding_mask = None,
                attn_mask = None): 
        """
        Parameters
        ----------
        x(float)[length, batch_size, embed_dim]
            Node features
        edge(long or int)[batch_size, length, length]
            Edge tokens (not embedded)
        key_padding_mask(float)[batch_size, length]
        attn_mask(bool)[length, length]:
            Assumed to be output of nn.Transformer.generate_square_subsequent_mask(Length)[:length, :length] 
        
        """
        
        # set up shape vars
        num_heads = self.num_heads
        length, bsz, embed_dim = x.shape
        head_dim = embed_dim // num_heads
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1)

        # prep attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.uint8:
                key_padding_mask = key_padding_mask.to(torch.bool)

            key_padding_mask = key_padding_mask.view(bsz, 1, 1, length).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, length)
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # prep edge feature
        edge = self.edge_embedding(edge).permute(0, 3, 1, 2).contiguous().view(bsz * num_heads, length, length)

        q = q / math.sqrt(q.shape[-1])
        attn_weights = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn_weights += attn_mask
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        ax.hist(attn_weights.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21))
        ax.hist(edge.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21), alpha=0.5)
        attn_weights += edge
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        attn_output = torch.bmm(attn_weights, v)
        
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class Layer(nn.Module):
    def __init__(self, d_model: int, nhead: int, edge_voc_size, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = GraphAttention(d_model, nhead, edge_voc_size, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        layer_norm_eps = 1e-5
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
    def forward(self, src, edge, src_mask = None,
                src_key_padding_mask = None):
        x = src
        x = x + self._sa_block(self.norm1(x), edge, src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
    def _sa_block(self, x, edge, attn_mask, key_padding_mask):
        x= self.self_attn(x, edge,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class GraphEncoder(nn.Module):
    def __init__(self, layer, n_layer, norm=None, init=dict()):
        """
        TransformerEncoderとほぼ同じ。
        
        Parameters
        ----------
        layer: dict
            Parameters for SelfAttentionLayer
        n_layer: int
        norm: dict or None
            Parameters for nn.LayerNorm
        init: dict
            Initialization for each name in self.encoder.layers[i].state_dict()
        """
        super().__init__()
        d_model = layer['d_model']
        self.layers = nn.ModuleList([Layer(**layer) 
            for i in range(n_layer)])
        if norm is not None:
            self.norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        else:
            self.norm = None

        # weight init
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    init_config2func(config)(param)

    def forward(self, nodes: torch.Tensor, node_padding_mask: torch.Tensor, bonds=torch.Tensor):
        """
        Parameters
        ----------
        nodes(torch.tensor(float))[batch_size, node_size, d_model]:
            node features.
        node_padding_mask(torch.tensor(torch.float))[batch_size, node_size]:
            Padding mask for nodes.
        bonds(torch.tensor(long))[batch_size, node_size, node_size]:
            Bond types before embedding.

        Returns
        -------
        output(torch.tensor(float))[batch_size, node_size, embed_size]:
            Output of encoder
        """
        output = nodes.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, bonds, src_mask=None, src_key_padding_mask=node_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        output = output.transpose(0, 1)
        return output

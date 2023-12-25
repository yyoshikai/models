"""
attn_mask is None, key_padding_mask is not Noneの時合ってる?
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models2 import init_config2func, function_name2func
from .. import module_type2class

import matplotlib.pyplot as plt


class GraphAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, edge_voc_size, edge_pad_token, edge_post_softmax, dropout=0,
            edge_embedded=False):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.edge_embedded = edge_embedded
        if self.edge_embedded:
            self.edge_embedding = nn.Linear(edge_voc_size, num_heads)
        else:
            self.edge_embedding = nn.Embedding(edge_voc_size, num_heads, padding_idx=edge_pad_token)
        self.edge_post_softmax = edge_post_softmax

    def forward(self, x: torch.Tensor, edge: torch.Tensor,
                key_padding_mask = None,
                attn_mask = None): 
        """
        Parameters
        ----------
        x(float)[length, batch_size, embed_dim]
            Node features
        edge(long or int)[batch_size, length, length] if not edge_embedded,
            edge(float)[batch_size, length, legnth, edge_voc_size] if edge_embedded
            Edge tokens
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

        """Attention weight check
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        ax.hist(attn_weights.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21))
        ax.hist(edge.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21), alpha=0.5)
        """
        
        # 231005追加 互換性はある。
        if self.edge_post_softmax:
            attn_weights = F.softmax(attn_weights, dim=-1) + edge
        else:
            attn_weights = F.softmax(attn_weights+edge, dim=-1)
        
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        attn_output = torch.bmm(attn_weights, v)
        
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, edge_voc_size, edge_pad_token, dim_feedforward: int = 2048, dropout: float = 0.1, 
            edge_post_softmax = False):
        super().__init__()
        self.self_attn = GraphAttention(d_model, nhead, edge_voc_size, 
            edge_pad_token=edge_pad_token, dropout=dropout, edge_post_softmax=edge_post_softmax)
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
        パラメータはTransformerEncoderとほぼ同じ。
        
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
        self.layers = nn.ModuleList([GraphAttentionLayer(**layer) 
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
        output(torch.tensor(float))[node_size, batch_size, embed_size]:
            Output of encoder
        """
        output = nodes.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, bonds, src_mask=None, src_key_padding_mask=node_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class MultiheadAttentionWithWeight(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None): 
        
        num_heads = self.num_heads
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // num_heads
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        
        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
        
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        src_len = k.size(1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights
    
class GraphMemoryDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu',
        layer_norm_eps=1e-5):
        
        super().__init__()
        self.self_attn = MultiheadAttentionWithWeight(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
        
        x = tgt
        x0, weight = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
        x = x + x0
        x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
        x = x + self._ff_block(self.norm3(x))
        return x, weight
    
    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, weight = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x), weight

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class GraphMemoryDecoder(nn.Module):
    """
    メモリからノード特徴量とエッジ行列を出力
    """
    def __init__(self, layer, n_layer, max_len):
        """
        """
        super().__init__()
        
        # Positional encoding
        d_model = layer['d_model']
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.xavier_uniform_(self.pe)

        self.layers = nn.ModuleList([ GraphMemoryDecoderLayer(**layer) for i in range(n_layer)])
    
    def forward(self, memory, padding_mask, memory_padding_mask):
        """
        Parameters
        ----------
        memory[input_len, batch_size, d_model]
        padding_mask[batch_size, output_len]
        memory_padding_mask[batch_size, input_len]

        Returns
        -------
        nodes[length, batch_size, d_model]
        edges[batch_size, length, length, nhead*n_layer]
        """
        _, batch_size, _ = memory.shape
        _, output_len = padding_mask.shape
        x = self.pe[:output_len].view(1, output_len, -1).expand(batch_size, -1, -1)
        weights = []
        for layer in self.layers:
            """
            def forward(self, tgt, memory, tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
            """
            x, weight = layer(tgt=x, memory=memory, tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=memory_padding_mask)
            # weight: [batch_size, nhead, output_len, output_len]
            weights.append(weight)
        weights = torch.cat(weights, dim=1).permute(0, 2, 3, 1)
        return x, weights

class AtomEmbedding(nn.Module):
    def __init__(self, output_size, atom_token_size, atom_pad_token=0, 
            chiral_token_size=4, chiral_pad_token=0):
        super().__init__()
        self.atom_type_embedding = nn.Embedding(num_embeddings=atom_token_size,
            embedding_dim=output_size, padding_idx=atom_pad_token, )
        self.chiral_embedding = nn.Embedding(num_embeddings=chiral_token_size, 
            embedding_dim=output_size, padding_idx=chiral_pad_token,)
        self.coord_embedding = nn.Linear(3, output_size)

    def forward(self, atom_types, chirals, coordinates):
        """
        Parameters
        ----------
        atom_types(long)[batch_size, length]:
        chirals(long)[batch_size, length]
        coordinates(float)[batch_size, 3]        
        """
        atom_types = self.atom_type_embedding(atom_types)
        chirals = self.chiral_embedding(chirals)
        coordinates = self.coord_embedding(coordinates)

        """check weight distribution
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        for i, (name, var) in enumerate(zip(['atom type', 'chiral', 'coordinates'],
                [atom_types, chirals, coordinates])):
            counts = np.histogram(var.detach().cpu().numpy().ravel(), range=(-5, 5), bins=20)[0]
            ax.bar(np.arange(-5, 5, 0.5)+(i+0.5)/6, counts, label=name, width=1/6)
        ax.legend()
        """
        return atom_types+chirals+coordinates

class NoiseAugmentor(nn.Module):
    def __init__(self, voc_size, p_smooth=0.9):
        super().__init__()
        self.voc_size = voc_size
        self.factor = np.log((p_smooth/(1-p_smooth))*(voc_size-1))

    def forward(self, input, noise_ratio=0.2):
        """
        input: 
        
        """
        input = F.one_hot(input, num_classes=self.atom_voc_size)
        input += torch.randn_like(input)*noise_ratio
        input = F.softmax(input*self.factor, dim=-1)
        return input

class RotationAugmentor(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.rstate = np.random.RandomState(seed=seed)

    def forward(self, coordinates):
        """
        Parameters
        ----------
        coordinates(float)[batch_size, length, 3]
        """

        rotation = np.linalg.qr(self.rstate.randn(3, 3))[0]
        if np.linalg.det(rotation) < 0:
            rotation[0] = -rotation[0]
        rotation = torch.tensor(rotation, device=coordinates.device)
        coordinates = torch.matmul(coordinates, rotation)
        return coordinates
    
# TransformerEncoderLayer相当
class DescriminatorLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0., activation='relu',
            layer_norm_eps=1e-5):
        """"""
        self.self_attn = GraphAttentionLayer()
        self.norm1 = nn.LayerNorm(eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.cross_attn = nn.MultiheadAttention()
        self.norm2 = nn.LayerNorm(eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = function_name2func[activation]
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(eps=layer_norm_eps)

        

    def forward(self, nodes0, edges0, nodes1, edges1, padding_mask):
        """
        nodes0(float)[length, batch_size, d_model]
        edges0(float)[batch_size, length, length, edge_voc_size]

        nodes1(float)[length, batch_size, d_model]
        edges1(float)[batch_size, length, length, edge_voc_size]


        """

        x0 = nodes0 + self._sa_block(self.norm1(nodes0), edges0, padding_mask)
        x1 = nodes1 + self._sa_block(self.norm1(nodes1), edges1, padding_mask)
        y0 = x0 + self._ca_block(self.norm2(x0), self.norm3(x1), padding_mask)
        y1 = x1 + self._ca_block(self.norm2(x1), self.norm3(x0), padding_mask)
        y0 = y0 + self._ff_block(self.norm3(y0))
        y1 = y1 + self._ff_block(self.norm3(y1))
        return y0, y1

    # self-attention block
    def _sa_block(self, x, edges, padding_mask):
        x, weight = self.self_attn(x, x, x, edges,
                           key_padding_mask=padding_mask)
        return self.dropout1(x), weight

    # multihead attention block
    def _ca_block(self, x, mem, key_padding_mask):
        x = self.cross_attn(x, mem, mem,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MoleculeDescriminator(nn.Module):
    def __init__(self, d_model, layer, n_layer, atom_voc_size, pooler, chiral_voc_size=4):
        """
        """
        super().__init__()
        self.layers = nn.ModuleList([DescriminatorLayer(**layer) for i in range(n_layer)])
        self.atom_linear = nn.Linear(atom_voc_size, d_model)
        self.chiral_linear = nn.Linear(chiral_voc_size, d_model)
        self.coord_linear = nn.Linear(3, d_model)

    def forward(self, atoms0, chirals0, coordinates0, bonds0,
            atoms1, chirals1, coordinates1, bonds1, 
            padding_mask):
        """
        0が1を参照して最終的に判断する。
        
        Parameters
        ---------
        atoms0(float)[batch_size, length, atom_voc_size]:
        
        chirals0(float)[batch_size, length, chiral_voc_size]:

        bonds0(float)[batch_size, length, length, bond_voc_size]:
        
        ~1についても同様。

        Returns
        -------
        x0(float)[length, batch_size, d_model]
        """
        x0 = self.atom_linear(atoms0)+self.chiral_linear(chirals0)+self.coord_linear(coordinates0)
        x0 = x0.transpose(0, 1)
        x1 = self.atom_linear(atoms1)+self.chiral_linear(chirals1)+self.coord_linear(coordinates1)
        x1 = x1.transpose(0, 1)
        for layer in self.layers:
            x0, x1 = layer(x0, bonds0, x1, bonds1, padding_mask)
        return x0

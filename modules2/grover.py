"""
The basic building blocks in model.
"""
import math

import numpy
import scipy.stats as stats
import torch
from torch import nn as nn
from torch.nn import LayerNorm, functional as F
from torch.nn.utils.rnn import pad_sequence

from ..datasets.grover import ATOM_FDIM, BOND_FDIM

def get_activation_function(activation: str):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target

def select_neighbor_and_aggregate(feature, index):
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(dim=1)


class MPNEncoder(nn.Module):
    def __init__(self, activation,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 input_layer="fc",
                 dynamic_depth='none'
                 ):
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(activation)

        # Input
        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None
                ):

        # Input
        if self.input_layer == 'fc':
            input = self.W_i(init_messages)  # num_bonds x hidden_size # f_bond
            message = self.act_func(input)  # num_bonds x hidden_size
        elif self.input_layer == 'none':
            input = init_messages
            message = input

        attached_fea = init_attached_features  # f_atom / f_bond

        # dynamic depth
        # uniform sampling from depth - 1 to depth + 1
        # only works in training.
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = numpy.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth

        # Message passing
        for _ in range(ndepth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2

            nei_message = select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            message = self.W_h(message)

            # BUG here, by default MPNEncoder use the dense connection in the message passing step.
            # The correct form should if not self.dense
            if self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        output = message

        return output  # num_atoms x hidden

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, outputs):
        if inputs is None:
            return self.dropout(self.norm(outputs))
        return inputs + self.dropout(self.norm(outputs))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=False):
        
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Linear(d_model, d_model, bias)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Head(nn.Module):
    def __init__(self, activation, bias, depth, dropout, undirected, dense, hidden_size, atom_messages=False):
        super().__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim

        # Here we use the message passing network as query, key and value.
        encoder_args = dict(activation=activation,
            atom_messages=atom_messages,
            init_message_dim=init_message_dim,
            attached_fea_fdim=attached_fea_dim,
            hidden_size=hidden_size,
            bias=bias,
            depth=depth,
            dropout=dropout,
            undirected=undirected,
            dense=dense,
            aggregate_to_atom=False,
            attach_fea=False,
            input_layer="none",
            dynamic_depth="truncnorm")
        self.mpn_q = MPNEncoder(**encoder_args)
        self.mpn_k = MPNEncoder(**encoder_args)
        self.mpn_v = MPNEncoder(**encoder_args)

    def forward(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a
            b2a = b2a
            b2revb = b2revb

        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v


class MTBlock(nn.Module):
    """
    The Multi-headed attention block.
    """

    def __init__(self,
                 head,
                 num_attn_head,
                 input_dim,
                 hidden_size,
                 activation="ReLU",
                 dropout=0.0,
                 bias=True,
                 atom_messages=False,
                 res_connection=False):
        super(MTBlock, self).__init__()
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

        self.W_i = nn.Linear(self.input_dim, self.hidden_size, bias=bias)
        self.attn = MultiHeadedAttention(h=num_attn_head,
                                         d_model=self.hidden_size,
                                         bias=bias,
                                         dropout=dropout)
        self.W_o = nn.Linear(self.hidden_size * num_attn_head, self.hidden_size, bias=bias)
        self.sublayer = SublayerConnection(self.hidden_size, dropout)
        for _ in range(num_attn_head):
            self.heads.append(Head(**head, atom_messages=atom_messages))

    def forward(self,  f_atoms, f_bonds, a2b, b2a, b2revb, a2a):

        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))

        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))

        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        x_out = self.attn(queries, keys, values)  # multi-headed attention
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
            return f_atoms
        else:
            f_bonds = self.sublayer(x_in, x_out)
            return f_bonds

class GTransEncoder(nn.Module):
    def __init__(self,
                 head,
                 hidden_size,
                 edge_fdim=BOND_FDIM,
                 node_fdim=ATOM_FDIM,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 bias=False):
        super(GTransEncoder, self).__init__()

        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = hidden_size
                node_input_dim_i = hidden_size
            block_args = dict(head=head,
                num_attn_head=num_attn_head,
                hidden_size=hidden_size,
                activation=activation,
                dropout=dropout,
                bias=bias)
            self.edge_blocks.append(MTBlock(**block_args,
                                            input_dim=edge_input_dim_i,
                                            atom_messages=False))
            self.node_blocks.append(MTBlock(**block_args,
                                            input_dim=node_input_dim_i,
                                            atom_messages=True))

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a2a):

        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            f_atoms = nb(f_atoms, original_f_bonds, a2b, b2a, b2revb, a2a)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            f_bonds = eb(original_f_atoms, f_bonds, a2b, b2a, b2revb, a2a)

        return f_atoms, f_bonds

class GroverEncoder(nn.Module):
    def __init__(self, head, hidden_size, memory_size, dropout, activation, 
            num_mt_block, num_attn_head, bias):
        super().__init__()
        self.grover = GTransEncoder(
            head=head,
            hidden_size=hidden_size,
            edge_fdim=BOND_FDIM, node_fdim=ATOM_FDIM,
            dropout=dropout, activation=activation,
            num_mt_block=num_mt_block, num_attn_head=num_attn_head,
            bias=bias)
        self.memory_linear = nn.Linear(hidden_size, memory_size, bias=False)
    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a):
        atom_output, bond_output = self.grover(f_atoms, f_bonds, a2b, b2a, b2revb, a2a) # [N_a_total, D], [N_b_total, D]
        device = atom_output.device

        graph_f_atoms = atom_output[1:].split([asc[1] for asc in a_scope], dim=0) # [B, ~N_a, D]
        graph_f_bonds = bond_output[1:].split([bsc[1] for bsc in b_scope], dim=0) # [B, ~N_b, D]
        graph_f = [torch.cat([fa, fb]) for fa, fb in zip(graph_f_atoms, graph_f_bonds)] # [B, ~N, D]
        padding_mask = [torch.full((len(f), ), fill_value=False, dtype=torch.bool, device=device) for f in graph_f]
        graph_f = pad_sequence(graph_f, batch_first=False, padding_value=0) # [N_max, B, D]
        graph_f = self.memory_linear(graph_f)        
        padding_mask = pad_sequence(padding_mask, batch_first=False, padding_value=True).T
        
        # layernormがあった方がよい?

        return graph_f, padding_mask

# Grover+Unimolのモデル用。
# GTransEncoderの出力をUnimolの入力に整形する。
# 通常のunimolのembeddingと合わせたいが, 原子の順番の対応がつくか分からないため保留
class Grover2UnimolEmbedding(nn.Module):
    def __init__(self,
            nhead, 
            d_model,  
            atom_size, 
            bond_size,
            atom_voc_size,
            bond_voc_size,
            bond_pad_token=0):
        super().__init__()
        self.f_atoms_emb = nn.Linear(atom_size, d_model)
        self.f_bonds_emb = nn.Linear(bond_size, nhead)
        # atom embedding
        self.atom_voc_size = atom_voc_size


        # atom pair embedding
        self.bond_pad_token = bond_pad_token
        self.bond_embedding = nn.Embedding(bond_voc_size, nhead, padding_idx=bond_pad_token)
        
    def forward(self,
            f_atoms: torch.FloatTensor,
            f_bonds: torch.FloatTensor,
            f_atoms_out: torch.FloatTensor, 
            f_bonds_out:torch.FloatTensor,
            b2a, b2revb,  
            a_scope: list, b_scope: list):
        """
        Parameters
        ----------
        f_atoms: torch.FloatTensor[Na_total, Dain]
        f_bonds: torch.FloatTensor[Nb_total, Dain]
        f_atoms_out: torch.FloatTensor[Na_total, Daout]
        f_bonds_out: torch.FloatTensor[Nb_total, Daout]
        a_scope: list[B, 2]
            [(start point, Na)]
        b_scope: list[B, 2]
            [(start point, Nb)]
        """

        # atom embedding
        device = f_atoms.device
        atoms_emb = torch.cat([f_atoms, f_atoms_out], dim=1) # [Na_total, Dain+Daout]
        atoms_emb = self.f_atoms_emb(atoms_emb) # [Na_total, Dm]
        atoms_emb = atoms_emb[1:].split([asc[1] for asc in a_scope], dim=0) # [B, ~Na, Dm]
        atoms_emb = pad_sequence(atoms_emb, batch_first=False, padding_value=0) # [Na, B, Dm]
        n_atom, batch_size, _ = atoms_emb.shape

        # bond embedding
        bonds_emb_g = torch.cat([f_bonds, f_bonds_out], dim=1) # [Nb_total, Dbin+Dbout]
        bonds_emb_g = self.f_bonds_emb(bonds_emb_g) # [Nb_total, Nh]
        _, nhead = bonds_emb_g.shape

        indices = torch.tensor([
            b2a[ib] + b2a[b2revb[ib]] * n_atom
            for ib in range(batch_size)], device=device, dtype=torch.long)
        for ib, (start_point, nb) in enumerate(b_scope):
            indices[start_point:start_point+nb] += ib * (n_atom**2)
        indices.unsqueeze_(-1)
        indices = indices.expand(-1, nhead)
        bonds_emb = torch.zeros((batch_size*n_atom*n_atom, nhead), dtype=torch.float, device=device)
        bonds_emb = torch.scatter(bonds_emb, dim=0, index=indices, src=bonds_emb)
        apairs = bonds_emb.reshape(batch_size, n_atom, n_atom, nhead)

        ## key_padding_mask
        padding_mask = [torch.full((asc[1], ), fill_value=False, dtype=torch.bool, device=device)
            for asc in a_scope]
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        apairs.masked_fill_(padding_mask.view(batch_size, 1, n_atom, 1), -torch.inf)

        apairs = apairs.permute(0, 3, 1, 2)
        
        return atoms_emb, apairs, padding_mask
"""
grover.pyからGTransEncoderに使わないものを除外
"""
import torch
from torch import nn as nn
from .grover import get_activation_function

class SelfAttention(nn.Module):
    def __init__(self, *, hidden, in_feature, out_feature):
        super(SelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn


class Readout(nn.Module):
    def __init__(self,
                 rtype: str = "none",
                 hidden_size: int = 0,
                 attn_hidden: int = None,
                 attn_out: int = None,
                 ):
        super(Readout, self).__init__()
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.rtype = "mean"

        if rtype == "self_attention":
            self.attn = SelfAttention(hidden=attn_hidden,
                                      in_feature=hidden_size,
                                      out_feature=attn_out)
            self.rtype = "self_attention"

    def forward(self, embeddings, scope):
        # Readout
        mol_vecs = []
        self.attns = []
        for _, (a_start, a_size) in enumerate(scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = embeddings.narrow(0, a_start, a_size)
                if self.rtype == "self_attention":
                    cur_hiddens, attn = self.attn(cur_hiddens)
                    cur_hiddens = cur_hiddens.flatten()
                    # Temporarily disable. Enable it if you want to save attentions.
                    # self.attns.append(attn.cpu().detach().numpy())
                else:
                    cur_hiddens = cur_hiddens.sum(dim=0) / a_size
                mol_vecs.append(cur_hiddens)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        return mol_vecs

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="PReLU", dropout=0.1, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)

    def forward(self, x):
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))

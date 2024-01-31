"""
[備考]
- q[l]の更新式を論文から変えている(論文の方がややおかしい気がする)

240110 UnimolPELayerを選択可とし, bdistをモデルの入力に追加した。

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models2 import function_name2func
from .tunnel import Tunnel
from .sequence import PositionalEncoding

class _UnimolLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff_factor=4, dropout=0., 
        activation='gelu', edge_layers=[]):
        super().__init__()
        
        # self attention layer
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.p_dropout = dropout
        
        # feed-forward layer
        dim_feedforward = int(embed_dim*d_ff_factor)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        layer_norm_eps = 1e-5
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]
        self.edge_tunnel = Tunnel(layers=edge_layers, 
            input_size=['batch_size', 'length', 'length', num_heads])

    def forward(self, x, edge, bdist=None):
        """
        Parameters
        ----------
        x(float)[length, batch_size, embed_dim]
        edge(float)[batch_size*num_heads, length, length]

        B: batch_size
        L: length
        Dh: d_model
        D': head_dim
        H: num_heads

        """
        
        # set up shape vars
        num_heads = self.num_heads
        length, bsz, embed_dim = x.shape
        head_dim = embed_dim // num_heads
        
        # residual connection
        x_res = x

        # pre layer_norm
        x = self.norm1(x)

        # attention
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # [B*H, L, L]

        """Attention weight check
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        ax.hist(attn_weights.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21))
        ax.hist(edge.detach().cpu().numpy().ravel(), bins=np.linspace(-2, 2, 21), alpha=0.5)
        """
        attn_weights = F.softmax(t + edge, dim=-1)
        if self.p_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        x = attn_output
        x = x_res+self.dropout1(x)

        # feed-forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))

        # edge feed-forward
        t = t.reshape(bsz, num_heads, length, length).transpose(1, 3)
        edge = self.edge_tunnel(t).transpose(1, 3).reshape(bsz*num_heads, length, length) + edge

        return x, edge

class _UnimolPELayer(nn.Module):
    def __init__(self, embed_dim, num_heads, sup_bdist, d_ff_factor=4, dropout=0., 
        activation='gelu', edge_layers=[]):
        """
        _UnimolLayerに相対的な位置の情報を追加
        
        
        """
        super().__init__()
        
        # self attention layer
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.p_dropout = dropout
        self.pe_emb = nn.Parameter(torch.randn((sup_bdist, embed_dim)))
        
        # feed-forward layer
        dim_feedforward = int(embed_dim*d_ff_factor)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        layer_norm_eps = 1e-5
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]
        self.edge_tunnel = Tunnel(layers=edge_layers, 
            input_size=['batch_size', 'length', 'length', num_heads])

    def forward(self, x, edge, bdist):
        """
        Parameters
        ----------
        x(float)[L, B, D]
        edge(float)[B*H, L, L]
        bdist(long)[L, L, B]
        
        B: batch_size
        L: length
        D: d_model
        Dh: head_dim
        H: num_heads
        P: sup_bdist(bdist)

        """
        
        # set up shape vars
        num_heads = self.num_heads
        length, bsz, embed_dim = x.shape
        head_dim = embed_dim // num_heads
        
        # residual connection
        x_res = x

        # pre layer_norm
        x = self.norm1(x)

        # attention
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # [L, B, H*Dh]
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) # [B*H, L, L]

        pe_qemb, pe_kemb, _ = self.in_proj(self.pe_emb).chunk(3, dim=-1) # [P, H*Dh]

        peq = F.embedding(bdist, weight=pe_qemb) # [L, L, B, H*Dh]
        pek = F.embedding(bdist, weight=pe_kemb) # [L, L, B, H*Dh]
        peq = peq.view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
        pek = pek.view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]

        t += torch.sum(peq*k.unsqueeze(1), dim=-1) + torch.sum(q.unsqueeze(2)*pek, dim=-1)
        t = t / math.sqrt(head_dim) 

        attn_weights = F.softmax(t + edge, dim=-1)
        if self.p_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        x = attn_output
        x = x_res+self.dropout1(x)

        # feed-forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))

        # edge feed-forward
        t = t.reshape(bsz, num_heads, length, length).transpose(1, 3)
        edge = self.edge_tunnel(t).transpose(1, 3).reshape(bsz*num_heads, length, length) + edge

        return x, edge

class _UnimolPEEELayer(nn.Module):
    """
    _UnimolPELayerを元に, エッジ情報もPEのような感じで入れるようにする。
    
    """
    
    def __init__(self, embed_dim, num_heads, sup_bdist, edge_voc_size, d_ff_factor=4, dropout=0., 
        activation='gelu', edge_layers=[]):
        super().__init__()
        
        # self attention layer
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.p_dropout = dropout
        self.pe_emb = nn.Parameter(torch.randn((sup_bdist, embed_dim)))

        self.ee_emb = nn.Parameter(torch.randn((edge_voc_size, embed_dim)))
        
        # feed-forward layer
        dim_feedforward = int(embed_dim*d_ff_factor)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        layer_norm_eps = 1e-5
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]
        self.edge_tunnel = Tunnel(layers=edge_layers, 
            input_size=['batch_size', 'length', 'length', num_heads])

    def forward(self, x, edge, bdist, edge_token):
        """
        Parameters
        ----------
        x(float)[L, B, D]
        edge(float)[B*H, L, L]
        bdist(long)[L, L, B]
        edge_token(long)[L, L, B]

        B: batch_size
        L: length
        D: d_model
        Dh: head_dim
        H: num_heads
        P: sup_bdist(bdist)

        """
        
        # set up shape vars
        num_heads = self.num_heads
        length, bsz, embed_dim = x.shape
        head_dim = embed_dim // num_heads
        
        # residual connection
        x_res = x

        # pre layer_norm
        x = self.norm1(x)

        # attention
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # [L, B, H*Dh]
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) # [B*H, L, L]

        pe_qemb, pe_kemb, _ = self.in_proj(self.pe_emb).chunk(3, dim=-1) # [P, H*Dh]
        ee_qemb, ee_kemb, _ = self.in_proj(self.ee_emb).chunk(3, dim=-1)

        peq = F.embedding(bdist, weight=pe_qemb) \
            .view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
        pek = F.embedding(bdist, weight=pe_kemb) \
            .view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
        eeq = F.embedding(edge_token, weight=ee_qemb) \
            .view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3)
        eek = F.embedding(edge_token, weight=ee_kemb) \
            .view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3)

        t += torch.sum((peq+eeq)*k.unsqueeze(1), dim=-1) + torch.sum(q.unsqueeze(2)*(pek+eek), dim=-1)
        t = t / math.sqrt(head_dim) 

        attn_weights = F.softmax(t + edge, dim=-1)
        if self.p_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        x = attn_output
        x = x_res+self.dropout1(x)

        # feed-forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))

        # edge feed-forward
        t = t.reshape(bsz, num_heads, length, length).transpose(1, 3)
        edge = self.edge_tunnel(t).transpose(1, 3).reshape(bsz*num_heads, length, length) + edge

        return x, edge

# Old version for compatibility (~231028)
# Embedding is included.
class UnimolEncoder(nn.Module):
    def __init__(self, 
            layer,
            n_layer,
            atom_voc_size, 
            bond_voc_size,
            apair_emb_size, 
            chiral_voc_size=4, 
            atom_pad_token=0,
            bond_pad_token=0,
            chiral_pad_token=0):
        super().__init__()
        print("[WARNING] use of UnimolEncoder is deprecated. Use UnimolEncoder2 instead.")
        # embedding
        d_model = layer['embed_dim']
        nhead = layer['num_heads']
        self.atom_voc_size = atom_voc_size
        self.atom_pad_token = atom_pad_token
        self.atype_embedding = nn.Embedding(atom_voc_size, d_model, padding_idx=atom_pad_token)
        self.chiral_embedding = nn.Embedding(chiral_voc_size, d_model, padding_idx=chiral_pad_token)
        self.apair_weight = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.apair_bias = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.means = nn.Parameter(torch.zeros((apair_emb_size, ), dtype=torch.float))
        self.stds = nn.Parameter(torch.ones((apair_emb_size, ), dtype=torch.float))
        self.bond_embedding = nn.Embedding(bond_voc_size, nhead, padding_idx=bond_pad_token)
        self.apair_linear = nn.Linear(apair_emb_size, nhead)
        self.layers = nn.ModuleList([_UnimolLayer(**layer) for _ in range(n_layer)])

    def forward(self, 
        atoms: torch.Tensor,
        chirals: torch.Tensor,
        coordinates: torch.Tensor,
        bonds: torch.Tensor):
        """
        Parameters
        ----------
        atoms(long)[batch_size, length]:
        chirals(long)[batch_size, length]
        coordinates(float)[batch_size, length, 3]  

        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs

        b: batch_size
        l: length
        apo: apair_emb_size
        """

        # embedding
        ## Atom embedding
        batch_size, length = atoms.shape
        atoms_emb = self.atype_embedding(atoms) + self.chiral_embedding(chirals)
        atoms_emb = atoms_emb.transpose(0, 1)

        ## Atom pair embedding
        ### distance
        distances = coordinates.unsqueeze(1) - coordinates.unsqueeze(2) # [b, l, l, 3]
        distances = torch.sqrt((distances**2).sum(dim=-1)) # [b, l, l]

        ### atom types
        apairs = atoms.unsqueeze(1)*self.atom_voc_size+ \
            atoms.unsqueeze(2) # [b, l, l]
        apair_weight = self.apair_weight(apairs) # [b, l, l, apo]
        apair_bias =  self.apair_bias(apairs)
        apairs = apair_weight*distances.unsqueeze(-1)+apair_bias # [b, l, l, apo]

        ### bond type
        bonds = self.bond_embedding(bonds)

        ### gaussian function
        stds = self.stds.abs() + 1e-5
        apairs = torch.exp(-0.5 * (((apairs - self.means) / stds) ** 2)) \
            / ((2*torch.pi)**0.5 * stds)
        
        apairs = self.apair_linear(apairs) + bonds # [B, L, L, H]

        ### key_padding_mask
        padding_mask = atoms == self.atom_pad_token
        apairs.masked_fill_(padding_mask.view(batch_size, 1, length, 1), -torch.inf)

        apairs = apairs.permute(0, 3, 1, 2).contiguous().view(-1, length, length) # [B*H, L, L]

        # layers
        for layer in self.layers:
            atoms_emb, apairs = layer(atoms_emb, apairs)
        apairs = apairs.view(batch_size, -1, length, length).permute(0, 2, 3, 1) # [B, L, L, H]

        return atoms_emb, apairs

class UnimolEmbedding(nn.Module):
    def __init__(self, 
            nhead, d_model,
            atom_voc_size, 
            bond_voc_size,
            apair_emb_size, 
            chiral_voc_size=4, 
            atom_pad_token=0,
            bond_pad_token=0,
            chiral_pad_token=0,
            nogauss: bool=False, 
            nochiral: bool=False):
        super().__init__()
        self.atom_voc_size = atom_voc_size
        self.atom_pad_token = atom_pad_token
        self.atype_embedding = nn.Embedding(atom_voc_size, d_model, padding_idx=atom_pad_token)
        self.chiral = not nochiral
        if self.chiral:
            self.chiral_embedding = nn.Embedding(chiral_voc_size, d_model, padding_idx=chiral_pad_token)
        self.apair_weight = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.apair_bias = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.gauss = not nogauss
        if self.gauss:
            self.means = nn.Parameter(torch.zeros((apair_emb_size, ), dtype=torch.float))
            self.stds = nn.Parameter(torch.ones((apair_emb_size, ), dtype=torch.float))
        self.bond_embedding = nn.Embedding(bond_voc_size, nhead, padding_idx=bond_pad_token)
        self.apair_linear = nn.Linear(apair_emb_size, nhead)

    def forward(self, 
        atoms: torch.Tensor,
        chirals: torch.Tensor,
        coordinates: torch.Tensor,
        bonds: torch.Tensor):
        """
        Parameters
        ----------
        atoms(long)[batch_size, length]:
        chirals(long)[batch_size, length]
        coordinates(float)[batch_size, length, 3]  

        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs

        b: batch_size
        l: length
        apo: apair_emb_size
        """

        # embedding
        ## Atom embedding
        batch_size, length = atoms.shape
        atoms_emb = self.atype_embedding(atoms)
        if self.chiral:  
            atoms_emb += self.chiral_embedding(chirals)
        atoms_emb = atoms_emb.transpose(0, 1)

        ## Atom pair embedding
        ### distance
        distances = coordinates.unsqueeze(1) - coordinates.unsqueeze(2) # [b, l, l, 3]
        distances = torch.sqrt((distances**2).sum(dim=-1)) # [b, l, l]

        ### atom types
        apairs = atoms.unsqueeze(1)*self.atom_voc_size+ \
            atoms.unsqueeze(2) # [b, l, l]
        apair_weight = self.apair_weight(apairs) # [b, l, l, apo]
        apair_bias =  self.apair_bias(apairs)
        apairs = apair_weight*distances.unsqueeze(-1)+apair_bias # [b, l, l, apo]

        ### bond type
        bonds = self.bond_embedding(bonds)

        ### gaussian function
        if self.gauss:
            stds = self.stds.abs() + 1e-5
            apairs = torch.exp(-0.5 * (((apairs - self.means) / stds) ** 2)) \
                / ((2*torch.pi)**0.5 * stds)
        
        apairs = self.apair_linear(apairs) + bonds # [B, L, L, H]

        ### key_padding_mask
        padding_mask = atoms == self.atom_pad_token
        apairs.masked_fill_(padding_mask.view(batch_size, 1, length, 1), -torch.inf)

        apairs = apairs.permute(0, 3, 1, 2)
        
        return atoms_emb, apairs

class UnimolGraphEmbedding(nn.Module):
    """
    Distance is ignored.
    """
    def __init__(self,
            nhead,
            d_model,
            atom_voc_size, 
            bond_voc_size,
            chiral_voc_size=4, 
            atom_pad_token=0,
            bond_pad_token=0,
            chiral_pad_token=0,
            posenc=None,
            shuffle_atoms=False,
            no_bond_emb=False,
            no_apair_emb=False):
        super().__init__()

        self.shuffle_atoms = shuffle_atoms
        # embedding
        self.nhead = nhead
        self.atom_voc_size = atom_voc_size
        self.atom_pad_token = atom_pad_token
        self.atype_embedding = nn.Embedding(atom_voc_size, d_model, padding_idx=atom_pad_token)
        self.chiral_embedding = nn.Embedding(chiral_voc_size, d_model, padding_idx=chiral_pad_token)
        self.apair_embedding = nn.Embedding(atom_voc_size**2, nhead)
        self.bond_embedding = nn.Embedding(bond_voc_size, nhead, padding_idx=bond_pad_token)
        if posenc is not None:
            self.posenc = PositionalEncoding(emb_size=d_model, **posenc)
        else:
            self.posenc = None
        self.no_bond_emb = no_bond_emb
        self.no_apair_emb = no_apair_emb

    def forward(self, 
        atoms: torch.Tensor,
        chirals: torch.Tensor,
        bonds: torch.Tensor):
        """
        Parameters
        ----------
        atoms(long)[batch_size, length]:
        chirals(long)[batch_size, length]
        bonds(long)[batch_size, length, length]

        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs

        b: batch_size
        l: length
        apo: apair_emb_size
        """
        batch_size, length = atoms.shape
        device = atoms.device

        # shuffle atoms
        if self.shuffle_atoms:
            idx = np.arange(length)
            np.random.shuffle(idx)
            idx = torch.tensor(idx, device=device, dtype=torch.long)
            atoms = atoms[:, idx]
            chirals = chirals[:, idx]
            bonds = bonds[:, idx][:, :, idx]

        # embedding
        ## Atom embedding
        atoms_emb = self.atype_embedding(atoms) + self.chiral_embedding(chirals)
        if self.posenc is not None:
            atoms_emb = self.posenc(atoms_emb)            
        else:
            atoms_emb = atoms_emb.transpose(0, 1)

        ## Atom pair embedding
        ### atom types
        if not self.no_apair_emb:
            apairs = atoms.unsqueeze(1)*self.atom_voc_size+ \
                atoms.unsqueeze(2) # [b, l, l]
            apairs =  self.apair_embedding(apairs) # [b, l, l, H]
        else:
            apairs = torch.zeros((batch_size, length, length, self.nhead),
                device=device, dtype=atoms_emb.dtype)

        ### bond type
        if not self.no_bond_emb:
            bonds = self.bond_embedding(bonds)
            apairs = apairs + bonds

        ### key_padding_mask
        padding_mask = atoms == self.atom_pad_token
        apairs.masked_fill_(padding_mask.view(batch_size, 1, length, 1), -torch.inf)

        apairs = apairs.permute(0, 3, 1, 2) # [B, H, L, L]

        return atoms_emb, apairs


unimol_layer_type2class = {
    'default': _UnimolLayer, 
    'pe': _UnimolPELayer, 
    'peee': _UnimolPEEELayer
}
class UnimolEncoder2(nn.Module):
    """
    Only layer processes.
    """
    def __init__(self, 
            layer: dict,
            n_layer):
        super().__init__()
        self.layer_type = layer.pop('type', 'default')
        layer_class = unimol_layer_type2class[self.layer_type]
        self.layers = nn.ModuleList([layer_class(**layer) for _ in range(n_layer)])

    def forward(self, 
        atoms_emb: torch.Tensor,
        apairs: torch.Tensor,
        bdist: torch.Tensor = None, 
        bonds: torch.Tensor = None):
        """
        Parameters
        ----------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, num_heads, length, length]
            Representation of atom pairs
        bdist(long)[batch_size, length, length]
            Distance matrix of graph
        bonds(long)[batch_size, length, length]
            Bond type of adjacency matrix
        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            
        B: batch_size
        L: length
        H: num_heads
        """
        batch_size, num_heads, length, _ = apairs.shape
        apairs = apairs.contiguous().view(batch_size*num_heads, length, length)
        if bdist is not None:
            bdist = bdist.permute(1,2,0).contiguous()
        if bonds is not None:
            bonds = bonds.permute(1,2,0).contiguous()
        # layers
        if self.layer_type == 'peee':
            for layer in self.layers:
                atoms_emb, apairs = layer(atoms_emb, apairs, bdist, bonds)
        else:
            for layer in self.layers:
                atoms_emb, apairs = layer(atoms_emb, apairs, bdist)
        apairs = apairs.view(batch_size, -1, length, length).permute(0, 2, 3, 1) # [B, L, L, H]

        return atoms_emb, apairs
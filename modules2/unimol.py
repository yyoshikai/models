"""
[備考]
- q[l]の更新式を論文から変えている(論文の方がややおかしい気がする)

240110 UnimolPELayerを選択可とし, bdistをモデルの入力に追加した。
240216 UnimolEmbeddingにlaplacian filter固有値のPEを追加可能にした(optionなので変更なし)
240217 UnimolEmbeddingにnh, is_aromatic, inringを追加可能にした。
240222 UnimolEmbeddingのapair_emb_typeを追加, 距離にのみgaussian関数をかけられるようにした。
240301 UnimolPELayerにedge_norm, edge_norm_weightを追加
240301 UnimolPELayerにedge_pe, edge_embed_dimを追加
240518 UnimolEncoderの出力のapairs, delta_apairsのpad領域を-torch.infではなく0とするように変更
    (現状UnimolEncoderの出力はpoolerにしか使っておらず, poolerではマスクしているので影響しないと思われる)
→ 240720 選択できるようにした。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .. import register_module
from ..models2 import function_name2func
from .tunnel import Tunnel
from .sequence import PositionalEncoding
from ..debug import count_nan

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
        activation='gelu',
        edge_layers=[],
        simple_pe=False, 
        edge_norm=False, edge_norm_weight=1.0, 
        edge_pe=False, edge_embed_dim=None):
        """
        _UnimolLayerに相対的な位置の情報を追加
        
        simple_pe: (240213追加) Trueの場合, pe_embを直接embeddingに使う。
        edge_norm, edge_norm_weight: 240301追加
            edge_norm = 'pre': edgeのresidual connectionの前にlayernorm
            edge_norm = 'post': edgeのresidual connectionの後にlayernorm
            edge_norm = False: 何もしない(通常)
        
        """
        super().__init__()
        
        # self attention layer
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.p_dropout = dropout

        # add edge info like PE
        self.edge_pe = edge_pe
        if self.edge_pe:
            self.edge_embed_dim = edge_embed_dim
            self.edge_in_proj = nn.Linear(edge_embed_dim, embed_dim*2)
            self.edge_out_proj = nn.Linear(num_heads, edge_embed_dim)

        # PE
        self.simple_pe = simple_pe
        if self.simple_pe:
            self.pe_qemb = nn.Parameter(torch.randn((sup_bdist, embed_dim)))
            self.pe_kemb = nn.Parameter(torch.randn((sup_bdist, embed_dim)))
        else:
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

        # edge normalization
        self.edge_norm_flag = edge_norm
        if self.edge_norm_flag:
            assert self.edge_norm_flag in ['post', 'pre']
            self.edge_norm = nn.LayerNorm(num_heads, eps=layer_norm_eps)
            nn.init.constant_(self.edge_norm.weight, edge_norm_weight)

    def forward(self, x: torch.Tensor, edge: torch.Tensor, bdist:torch.Tensor):
        
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

        # add PE
        if self.simple_pe:
            pe_qemb, pe_kemb = self.pe_qemb, self.pe_kemb
        else:
            pe_qemb, pe_kemb, _ = self.in_proj(self.pe_emb).chunk(3, dim=-1) # [P, H*Dh]
        peq = F.embedding(bdist, weight=pe_qemb) # [L, L, B, H*Dh]
        pek = F.embedding(bdist, weight=pe_kemb) # [L, L, B, H*Dh]
        peq = peq.view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
        pek = pek.view(length, length, bsz*num_heads, head_dim) \
            .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
        pe_t_k = torch.sum(peq*k.unsqueeze(1), dim=-1)
        pe_t_q = torch.sum(q.unsqueeze(2)*pek, dim=-1)
        t += pe_t_k + pe_t_q

        # add edge info
        if self.edge_pe:
            edge0 = edge
            edge = edge.view(bsz, self.edge_embed_dim, length, length).permute(2, 3, 0, 1) # [L, L, B, De]
            edge = self.edge_in_proj(edge) # [L, L, B, H*Dh*2]
            edge_k, edge_q = edge.chunk(2, dim=-1) # [L, L, B, H*Dh]
            edge_k = edge_k.contiguous() \
                .view(length, length, bsz*num_heads, head_dim) \
                .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
            edge_q = edge_q.contiguous() \
                .view(length, length, bsz*num_heads, head_dim) \
                .permute(2, 0, 1, 3) # [B*H, L, L, Dh]
            edge_t_k = torch.sum(edge_q*k.unsqueeze(1), dim=-1)
            edge_t_q = torch.sum(q.unsqueeze(2)*edge_k, dim=-1)
            t += edge_t_k + edge_t_q
            edge = edge0
        
        dedge = t
        t = t / math.sqrt(head_dim)
        if not self.edge_pe:
            t += edge
        attn_weights = F.softmax(t, dim=-1)
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
        edge_dim = self.edge_embed_dim if self.edge_pe else num_heads
        t = dedge
        t = t / math.sqrt(head_dim)
        if self.edge_pe:
            t = t.reshape(bsz, num_heads, length, length).transpose(1, 3)
            t = self.edge_out_proj(t)
            t = t.transpose(1, 3).reshape(bsz*edge_dim, length, length)
        if len(self.edge_tunnel) > 0:
            t = t.reshape(bsz, edge_dim, length, length).transpose(1, 3)
            t = self.edge_tunnel(t)
            t = t.transpose(1, 3).reshape(bsz*edge_dim, length, length)

        if self.edge_norm_flag == 'post':
            edge = edge + t
            
            edge = edge.reshape(bsz, edge_dim, length, length).transpose(1, 3)
            padding_mask = torch.isneginf(edge)
            edge.masked_fill_(padding_mask, 1)
            edge = self.edge_norm(edge)
            edge.masked_fill_(padding_mask, -torch.inf)
            edge = edge.transpose(1, 3).reshape(bsz*edge_dim, length, length)

        elif self.edge_norm_flag == 'pre':
            t = t.reshape(bsz, edge_dim, length, length).transpose(1, 3)
            t = self.edge_norm(t)
            t = t.transpose(1 ,3).reshape(bsz*edge_dim, length, length)

            edge = edge + t
        else:
            edge = edge + t

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
@register_module
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

@register_module
class UnimolEmbedding(nn.Module):
    def __init__(self, 
            nhead, 
            d_model,
            atom_voc_size, 
            bond_voc_size,
            apair_emb_size, 
            chiral_voc_size=4, 
            atom_pad_token=0,
            bond_pad_token=0,
            chiral_pad_token=0,
            nogauss: bool=False, 
            nochiral: bool=False, 
            lap_pe_size: int=0, lap_pe_factor: float=1.0,
            add_nh: bool=False, max_nh: int=None, add_is_aromatic: bool=False, add_inring: bool=False, max_inring: int=None, 
            bond_rdir_mask: bool=False, seed: int=0, 
            apair_emb_type='base',
            no_calc_coord=False):
        """
        Parameters
        ----------
        lap_pe_size, lap_pe_factor:
            ラプラシアン行列の固有ベクトルをPositional Encodingとして加えるかどうか, 
            及びその重み。
        add_nh, max_nh, add_is_aromatic, add_inring, max_inring:
            結合する水素数, 芳香族かどうか, 環の中にあるかどうか のフラグを入れるかどうか, 
            及びその入力の範囲
        bond_rdir_mask: 
            Trueの場合, 原子にランダムな順番を設定して, 小→大(逆?)の方向のみの隣接行列とする。
        apair_emb_type:
            'base': 通常のembedding
            'simple': (240222作成) gaussian関数を距離にのみ適用, 直接embeddingを求める。
        
            
        """
        super().__init__()

        self.atom_voc_size = atom_voc_size
        self.atom_pad_token = atom_pad_token
        self.rng = np.random.default_rng(seed=seed)
        
        # atom embedding
        self.atype_embedding = nn.Embedding(atom_voc_size, d_model, padding_idx=atom_pad_token)
        self.chiral = not nochiral
        self.chiral_pad_token = chiral_pad_token
        if self.chiral:
            self.chiral_embedding = nn.Embedding(chiral_voc_size, d_model, padding_idx=chiral_pad_token)
        
        ## laplacian positional embedding
        self.lap_pe_size = lap_pe_size
        if lap_pe_size > 0:
            self.lap_emb = nn.Linear(lap_pe_size, d_model)
            self.lap_pe_factor = lap_pe_factor
        
        ## atom information embedding
        self.add_nh = add_nh
        if self.add_nh:
            self.nh_emb = nn.Embedding(max_nh, d_model, padding_idx=0)
        self.add_is_aromatic = add_is_aromatic
        if self.add_is_aromatic:
            self.is_aromatic_emb = nn.Embedding(2, d_model, padding_idx=0)
        self.add_inring = add_inring
        if self.add_inring:
            self.max_inring = max_inring
            self.inring_emb = nn.Linear(max_inring, d_model, bias=False)
            nn.init.normal_(self.inring_emb.weight) # Compatible with nn.Embedding
        
        # atom pair embedding
        ## bond type
        self.bond_pad_token = bond_pad_token
        self.bond_embedding = nn.Embedding(bond_voc_size, nhead, padding_idx=bond_pad_token)
        self.bond_rdir_mask = bond_rdir_mask
        self.calc_coord = not no_calc_coord

        ## distance & atom type
        self.apair_emb_type = apair_emb_type
        self.apair_weight = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.apair_bias = nn.Embedding(atom_voc_size**2, apair_emb_size)
        self.apair_linear = nn.Linear(apair_emb_size, nhead)
        if self.apair_emb_type == 'base':
            self.gauss = not nogauss
        if not (self.apair_emb_type == 'base' and (not self.gauss)):
            self.means = nn.Parameter(torch.zeros((apair_emb_size, ), dtype=torch.float))
            self.stds = nn.Parameter(torch.ones((apair_emb_size, ), dtype=torch.float))

    def forward(self, 
        atoms: torch.Tensor,
        chirals: torch.Tensor,
        coordinates: torch.Tensor,
        bonds: torch.Tensor,
        nh: torch.Tensor=None,
        is_aromatic: torch.Tensor=None,
        inring: torch.Tensor=None):
        """
        Parameters
        ----------
        atoms(long)[batch_size, length]:
        chirals(long)[batch_size, length]
        coordinates(float)[batch_size, length, 3]
            or [batch_size, length, length] when no_calc_coord=True
        bonds(long)[batch_size, length, length]
        nh(long)[B, Na]
        is_aromatic(long)[B, Na]
        inring(float)[B, Na, max_inring]
        
        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs
        bonds(float)[B, Na, Na]

        b: batch_size
        l: length
        apo: apair_emb_size
        """
        
        batch_size, length = atoms.shape
        device = atoms.device

        # Atom embedding
        atoms_emb = self.atype_embedding(atoms) # [B, Na, D]
        if self.chiral:  
            atoms_emb += self.chiral_embedding(chirals)

        ## Laplacian positional embedding
        if self.lap_pe_size > 0:
            adj_whole = (bonds > 0).to(torch.float) # [B, Na, Na]
            lap_pe = []
            for ibatch in range(batch_size): # 重い?
                mask = atoms[ibatch] != self.atom_pad_token
                adj = adj_whole[ibatch][mask][:, mask] # [~Na, ~Na]
                n_atomi = len(adj)
                deg = torch.sum(adj, dim=0)
                lap = torch.eye(len(adj), device=device) - adj / torch.sqrt(deg) / torch.sqrt(deg).unsqueeze(-1) # [~Na, ~Nv]
                vs, vecs = torch.linalg.eig(lap)
                vs, vecs = vs.real, vecs.real
                vecs = vecs[:, torch.argsort(vs)[1:self.lap_pe_size+1]] # [~Na, ~Nv]
                n_vec = vecs.shape[1]
                if n_vec < self.lap_pe_size:
                    vecs = torch.cat([vecs, 
                        torch.zeros(n_atomi, self.lap_pe_size-n_vec, device=device)], dim=-1) # [~Na, Nv]
                lap_pe.append(vecs)
            lap_pe = pad_sequence(lap_pe, padding_value=0, batch_first=True) # [B, Na, Nv]
            lap_pe = self.lap_emb(lap_pe) # [B, Na, D]
            atoms_emb += lap_pe * self.lap_pe_factor

        ## Atom information embedding
        if self.add_nh:
            atoms_emb += self.nh_emb(nh)
        if self.add_is_aromatic:
            atoms_emb += self.is_aromatic_emb(is_aromatic)
        if self.add_inring:
            atoms_emb += self.inring_emb(inring)
        atoms_emb = atoms_emb.transpose(0, 1) # [Na, B, D]

        # Atom pair embedding
        ## bond type
        if self.bond_rdir_mask:
            random_order = np.arange(length, dtype=int)
            self.rng.shuffle(random_order)
            random_mask = (random_order[:, np.newaxis] > random_order[np.newaxis, :])
            random_mask = torch.tensor(random_mask).unsqueeze(0).expand(batch_size, -1, -1)
            
            bonds[random_mask] = self.bond_pad_token
        bonds = self.bond_embedding(bonds)

        ## distance
        if self.calc_coord:
            distances = coordinates.unsqueeze(1) - coordinates.unsqueeze(2) # [b, l, l, 3]
            distances = torch.sqrt((distances**2).sum(dim=-1)) # [b, l, l]
        else:
            distances = coordinates

        ## atom type & distance
        apairs = atoms.unsqueeze(1)*self.atom_voc_size+ \
            atoms.unsqueeze(2) # [B, Na, Na]
        apair_weight = self.apair_weight(apairs) # [B, Na, Na, Dapair]
        apair_bias =  self.apair_bias(apairs) # [B, Na, Na, Dapair]
            
        if self.apair_emb_type == 'base':
            apair_dist = apair_weight*distances.unsqueeze(-1)
            apairs = apair_dist+apair_bias # [B, Na, Na, Dapair]
            if self.gauss:
                stds = self.stds.abs() + 1e-5
                apairs = torch.exp(-0.5 * (((apairs - self.means) / stds) ** 2)) \
                    / ((2*torch.pi)**0.5 * stds)
        elif self.apair_emb_type == 'simple':
            ## distance embedding
            stds = self.stds.abs() + 1e-5
            dist_emb = torch.exp(-0.5*((distances.unsqueeze(-1) - self.means) / stds)**2) \
                    / ((2*torch.pi)**0.5*stds)
            apair_dist = apair_weight * dist_emb
            apairs = apair_dist + apair_bias
        else:
            raise ValueError(f"Unsupported apair_emb_type: {self.apair_emb_type}")
        apairs = self.apair_linear(apairs) + bonds # [B, L, L, H]

        ## key_padding_mask
        padding_mask = atoms == self.atom_pad_token
        apairs.masked_fill_(padding_mask.view(batch_size, 1, length, 1), -torch.inf)

        apairs = apairs.permute(0, 3, 1, 2)
        
        return atoms_emb, apairs

@register_module
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

# 240215ごろ? のUnimolEmbeddingからfork
# Edgeの情報もembeddingとして出力する。
@register_module
class UnimolETEmbedding(nn.Module):
    def __init__(self, 
            nhead, d_model,
            atom_voc_size, 
            bond_voc_size,
            apair_emb_size, 
            sup_bdist,
            chiral_voc_size=4, 
            atom_pad_token=0,
            bond_pad_token=0,
            chiral_pad_token=0,
            nogauss: bool=False, 
            nochiral: bool=False):
        super().__init__()
        self.nhead = nhead
        self.atom_voc_size = atom_voc_size
        self.atom_pad_token = atom_pad_token
        self.bond_pad_token = bond_pad_token
        self.sup_bdist = sup_bdist
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

        # For bond embedding
        self.btype_embedding = nn.Embedding(bond_voc_size, d_model, padding_idx=bond_pad_token)
        self.bond_atom_embedding = nn.Embedding(atom_voc_size, d_model, padding_idx=atom_pad_token)

        self.ab_connect_embedding = nn.Embedding(2, nhead, padding_idx=0)

    def forward(self, 
        atoms: torch.Tensor,
        chirals: torch.Tensor,
        coordinates: torch.Tensor,
        bonds: torch.Tensor, 
        bond_indices: torch.Tensor,
        bond_values: torch.Tensor, 
        bdist: torch.Tensor):
        """
        Parameters
        ----------
        atoms(long)[B, Na]
        chirals(long)[B, Na]
        coordinates(float)[B, Na, 3]
        bonds(long)[B, Na, Na]
        bond_indices(long)[B, Nb, 2]
        bond_values(long)[B, Nb]
        bdist(long)[B, Na, Na]

        Returns
        -------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs
        bonds_emb(float)[n_bond_max, batch_size, d_model]

        B: batch_size
        Na: Num of atoms
        Nb: Num of bonds
        apo: apair_emb_size
        """

        # embedding
        ## Node embedding
        ### Atom embedding
        batch_size, n_atom = atoms.shape
        device = atoms.device
        atoms_emb = self.atype_embedding(atoms) # [B, Na, D]
        if self.chiral:  
            atoms_emb += self.chiral_embedding(chirals)

        ### Bond embedding
        _, n_bond = bond_values.shape
        bonds_emb = self.btype_embedding(bond_values) # [B, Nb, D]
        bonds_atom0 = torch.gather(atoms, dim=1, index=bond_indices[:,:,0]) # [B, Nb]
        bonds_atom1 = torch.gather(atoms, dim=1, index=bond_indices[:,:,1]) # [B, Nb]
        bonds_emb += \
            self.bond_atom_embedding(bonds_atom0) \
            +self.bond_atom_embedding(bonds_atom1)
        
        atoms_emb = torch.cat([atoms_emb, bonds_emb], dim=1) # [B, Na+Nb, D]
        atoms_emb = atoms_emb.transpose(0, 1) # [Na+Nb, B, D]

        ## Node pair embedding
        ### atom-atom pairs
        #### distance
        distances = coordinates.unsqueeze(1) - coordinates.unsqueeze(2) # [B, Na, Na, 3]
        distances = torch.sqrt((distances**2).sum(dim=-1)) # [B, Na, Na]

        #### atom types
        apairs = atoms.unsqueeze(1)*self.atom_voc_size+ \
            atoms.unsqueeze(2) # [b, l, l]
        apair_weight = self.apair_weight(apairs) # [b, l, l, apo]
        apair_bias =  self.apair_bias(apairs)
        apairs = apair_weight*distances.unsqueeze(-1)+apair_bias # [b, l, l, apo]

        #### bond type
        bonds = self.bond_embedding(bonds)

        #### gaussian function
        if self.gauss:
            stds = self.stds.abs() + 1e-5
            apairs = torch.exp(-0.5 * (((apairs - self.means) / stds) ** 2)) \
                / ((2*torch.pi)**0.5 * stds)
        
        apairs = self.apair_linear(apairs) + bonds # [B, Na, Na, H]

        ### atom-node pairs
        abpairs_oh = F.one_hot(bond_indices[:, :, 0], num_classes=n_atom) \
            + F.one_hot(bond_indices[:, :, 1], num_classes=n_atom) # [B, Nb, Na]
        abpairs_oh = abpairs_oh * (bond_values != self.bond_pad_token).to(torch.int).unsqueeze(-1)
        abpairs = self.ab_connect_embedding(abpairs_oh) # [B, Nb, Na, H]

        ### concatenate
        apairs = torch.cat([
            torch.cat([apairs, abpairs], dim=1), # [B, Na+Nb, Na, H]
            torch.cat([abpairs.transpose(1,2), 
                torch.zeros((batch_size, n_bond, n_bond, self.nhead), 
                    device=device, dtype=torch.long)], dim=1) # [B, Nb, Na+Nb, H]
        ], dim=2)

        ### key_padding_mask
        padding_mask = torch.cat([
            atoms == self.atom_pad_token, # [B, Na]
            bond_values == self.bond_pad_token # [B, Nb]
        ], dim=1) # [B, Na+Nb]
        
        apairs.masked_fill_(padding_mask.view(batch_size, 1, n_atom+n_bond, 1), -torch.inf)
        apairs = apairs.permute(0, 3, 1, 2) # [B, H, Na+Nb, Na+Nb]

        ## bdist
        abpairs_oh = abpairs_oh * self.sup_bdist
        bdist = torch.cat([
            torch.cat([bdist, abpairs_oh], dim=1), # [B, Na+Nb, Na]
            torch.cat([abpairs_oh.transpose(1,2), 
                torch.zeros(batch_size, n_bond, n_bond, dtype=torch.long, device=device)], dim=1) # [B, Na+Nb, Nb]
        ], dim=2)
        
        return atoms_emb, apairs, bdist, padding_mask

unimol_layer_type2class = {
    'default': _UnimolLayer, 
    'pe': _UnimolPELayer, 
    'peee': _UnimolPEEELayer
}
@register_module
class UnimolEncoder2(nn.Module):
    """
    Only layer processes.
    """
    def __init__(self, 
            layer: dict,
            n_layer, 
            no_zero_fill_pad = False):
        super().__init__()
        self.layer_type = layer.pop('type', 'default')
        layer_class = unimol_layer_type2class[self.layer_type]
        self.layers = nn.ModuleList([layer_class(**layer) for _ in range(n_layer)])
        self.no_zero_fill_pad = no_zero_fill_pad

    def forward(self, 
        atoms_emb: torch.Tensor,
        apairs: torch.Tensor,
        bdist: torch.Tensor = None, 
        bonds: torch.Tensor = None,
        output_delta_apairs=False):
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
        input_apairs = apairs
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
        if not self.no_zero_fill_pad:
            apairs.masked_fill_(torch.isinf(apairs), 0.0)
        output = atoms_emb, apairs
        if output_delta_apairs:
            delta_apairs = apairs - input_apairs.permute(0, 2, 3, 1)
            delta_apairs.masked_fill_(~torch.isfinite(delta_apairs), 0.0) # inf - infを0埋め
            output += (delta_apairs, )
        return output

@register_module
class DummyAdder(nn.Module):
    def __init__(self, atom_dummy, 
            chiral_dummy=0, 
            distance_dummy=0.0,
            bdist_dummy=0,
            bond_dummy=0,
            nh_dummy=0,
            is_aromatic_dummy=0,
            inring_dummy=0,
            dist_eps=1e-9
            ):
        super().__init__()
        self.atom_dummy = atom_dummy
        self.chiral_dummy = chiral_dummy
        self.distance_dummy = distance_dummy
        self.bdist_dummy = bdist_dummy
        self.bond_dummy = bond_dummy
        self.nh_dummy = nh_dummy
        self.is_aromatic_dummy = is_aromatic_dummy
        self.inring_dummy = inring_dummy
        self.dist_eps = dist_eps
    
    def forward(self, atoms, chirals, coordinates, bdist, bonds, 
        nh=None, is_aromatic=None, inring=None):
        """
        Parameters
        ----------
        atoms(long)[B, L]:
        chirals(long)[B, L]
        coordinates(float)[B, L, 3]
        bdist(long)[B, L, L]
            Distance matrix of graph
        bonds(long)[B, L, L]
        nh(long)[B, Na]
        is_aromatic(long)[B, Na]
        inring(float)[B, Na, max_inring]
        """
        batch_size, length = atoms.shape
        device = atoms.device

        atoms = torch.cat([
            torch.full((batch_size, 1), fill_value=self.atom_dummy, dtype=atoms.dtype, device=device),
            atoms
        ], dim=1)

        chirals = torch.cat([
            torch.full((batch_size, 1), fill_value=self.chiral_dummy, dtype=chirals.dtype, device=device),
            chirals
        ], dim=1)

        distances = coordinates.unsqueeze(1) - coordinates.unsqueeze(2) # [b, l, l, 3]
        distances = torch.sqrt((distances**2).sum(dim=-1)) # [b, l, l]

        distances = torch.cat([
            torch.full((batch_size, 1, length), fill_value=self.distance_dummy, dtype=distances.dtype, device=device),
            distances
        ], dim=1) # [B, L+1, L]
        distances = torch.cat([
            torch.full((batch_size, length+1, 1), fill_value=self.distance_dummy, dtype=distances.dtype, device=device),
            distances
        ], dim=2) # [B, L+1, L+1]

        bdist = torch.cat([
            torch.full((batch_size, 1, length), fill_value=self.bdist_dummy, dtype=bdist.dtype, device=bdist.device),
            bdist
        ], dim=1)
        bdist = torch.cat([
            torch.full((batch_size, length+1, 1), fill_value=self.bdist_dummy, dtype=bdist.dtype, device=bdist.device),
            bdist
        ], dim=2)



        bonds = torch.cat([
            torch.full((batch_size, 1, length), fill_value=self.bond_dummy, dtype=bonds.dtype, device=bonds.device),
            bonds
        ], dim=1) # [B, L+1, L]
        bonds = torch.cat([
            torch.full((batch_size, length+1, 1), fill_value=self.bond_dummy, dtype=bonds.dtype, device=bonds.device),
            bonds
        ], dim=2)
        nh = torch.cat([
            torch.full((batch_size, 1), fill_value=self.nh_dummy, dtype=nh.dtype, device=nh.device),
            nh
        ], dim=1)
        is_aromatic = torch.cat([
            torch.full((batch_size, 1), fill_value=self.is_aromatic_dummy, dtype=is_aromatic.dtype, device=is_aromatic.device),
            is_aromatic
        ], dim=1)
        _, _, max_inring = inring.shape
        inring = torch.cat([
            torch.full((batch_size, 1, max_inring), fill_value=self.inring_dummy, dtype=inring.dtype, device=inring.device),
            inring
        ], dim=1)


        return atoms, chirals, distances, bdist, bonds, nh, is_aromatic, inring

# transposeについて

class _UnimolDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff_factor=4, dropout=0., 
        activation='gelu'):
        """
        ~240718 UnimolPELayerより派生。
        """
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

    def forward(self, x: torch.Tensor, edge: torch.Tensor):
        
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
        
        dedge = t
        t = t / math.sqrt(head_dim)
        t += edge
        attn_weights = F.softmax(t, dim=-1)
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
        t = dedge
        t = t / math.sqrt(head_dim)
        edge = edge + t

        return x, edge


@register_module
class UnimolDecoder(nn.Module):
    def __init__(self, 
            layer: dict,
            n_layer, ):
        super().__init__()
        self.layers = nn.ModuleList([_UnimolDecoderLayer(**layer) for _ in range(n_layer)])


    def forward(self, 
        atoms_emb: torch.Tensor,
        apairs: torch.Tensor):
        """
        Parameters
        ----------
        atoms_emb(float)[length, batch_size, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            Representation of atom pairs
        
        Returns
        -------
        atoms_emb(float)[batch_size, length, d_model]
        apairs(float)[batch_size, length, length, num_heads]
            
        B: batch_size
        L: length
        H: num_heads
        """
        apairs = apairs.permute(0, 3, 1, 2).contiguous()
        batch_size, num_heads, length, _ = apairs.shape
        apairs = apairs.view(batch_size*num_heads, length, length)
        
        for layer in self.layers:
            atoms_emb, apairs = layer(atoms_emb, apairs)
        apairs = apairs.view(batch_size, -1, length, length).permute(0, 2, 3, 1) # [B, L, L, H]
        apairs.masked_fill_(torch.isinf(apairs), 0.0)
        atoms_emb = atoms_emb.permute(1, 0, 2).contiguous() # [B, L, D]

        return  atoms_emb, apairs

@register_module
class Noiser(nn.Module):
    def __init__(self, voc_size, mu, sigma):
        super().__init__()
        self.voc_size = voc_size
        self.mu = mu
        self.sigma = sigma


    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: [*]
        
        Returns
        -------
        x: [*, voc_size]
        """
        x = F.one_hot(x, num_classes=self.voc_size).to(torch.float) * self.mu
        x = x + torch.randn_like(x, device=x.device) * self.sigma
        return F.softmax(x, dim=-1)

class _UnimolDescriminatorLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff_factor=4, dropout=0., activation='gelu'):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        layer_norm_eps = 1e-5
        
        # self attention layer
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.p_dropout = dropout

        # multihead attention
        self.mha_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mha_q_proj = nn.Linear(embed_dim, embed_dim)
        self.mha_kv_proj = nn.Linear(embed_dim, 2*embed_dim)
        self.mha_out_proj = nn.Linear(embed_dim, embed_dim)

        # feed-forward layer
        dim_feedforward = int(embed_dim*d_ff_factor)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]


    def forward(self, node0: torch.Tensor, edge0: torch.Tensor,
            node1: torch.Tensor, edge1: torch.Tensor):      
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
        length, bsz, embed_dim = node0.shape
        head_dim = embed_dim // num_heads
        
        # self attention
        ## residual connection
        x_res = node0

        ## pre layer_norm
        x = self.norm1(node0)

        ## attention
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # [L, B, H*Dh]
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # [B*H, L, L]
        sa_t = t
        t = t + edge0
        attn_weights = F.softmax(t, dim=-1)
        if self.p_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        ## residual connection
        x = attn_output
        x = x_res+self.dropout1(x)

        # multihead attention
        x_res = x
        x = self.mha_norm(x)

        q = self.mha_q_proj(x) # [L, B, H*Dh]
        k, v = self.mha_kv_proj(node1).chunk(2, dim=-1) # [L, B, H*Dh]
        q = q.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        k = k.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        v = v.contiguous().view(length, bsz * num_heads, head_dim).transpose(0, 1) # [B*H, L, Dh]
        t = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # [B*H, L, L]
        mha_t = t
        t = t + edge1
        attn_weights = F.softmax(t, dim=-1)
        if self.p_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.p_dropout)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)


        # feed-forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))

        edge0 = edge0 + sa_t + mha_t

        return x, edge0

@register_module
class UnimolDescriminator(nn.Module):
    def __init__(self, layer, n_layer, nhead, d_model,
             node_voc_size, edge_voc_size,
             node_pad_token=0):
        super().__init__()
        self.node_embedding = nn.Embedding(node_voc_size, d_model)
        self.edge_embedding = nn.Embedding(edge_voc_size, nhead)
        self.node_pad_token = node_pad_token

        self.layers0 = nn.ModuleList([_UnimolDescriminatorLayer(**layer) for _ in range(n_layer)])
        self.layers1 = nn.ModuleList([_UnimolDescriminatorLayer(**layer) for _ in range(n_layer)])
        self.predictor = nn.Linear(d_model, 1)

    def forward(self, node: torch.Tensor, edge: torch.Tensor,
            node_emb: torch.Tensor, edge_emb: torch.Tensor):
        """
        Parameters
        ----------
        node: int[B, L]
        edge: int[B, L, L]
        node_emb: float[B, L, Vn]
        edge_emb: float[B, L, L, Ve]
        """
        B, L, node_voc_size = node_emb.shape
        _, _, _, edge_voc_size = edge_emb.shape

        # Embedding
        ## 0
        node0 = self.node_embedding(node) # [B, L, D]
        edge0 = self.edge_embedding(edge) # [B, L, L, H]
        H = edge0.shape[3]
        
        ## 1
        node1 = torch.mm(node_emb.view(-1, node_voc_size), 
            self.node_embedding.weight) \
            .view(B, L, -1) # [B, L, D]
        edge1 = torch.mm(edge_emb.view(-1, edge_voc_size), 
            self.edge_embedding.weight) \
            .view(B, L, L, -1)
        node0 = node0.transpose(0, 1) # [L, B, D]
        node1 = node1.transpose(0, 1) # [L, B, D]

        ## Mask padding
        padding_mask = node == self.node_pad_token # [B, L]
        edge0.masked_fill_(padding_mask.view(B, 1, L, 1), -torch.inf) # [B, L, L, H]
        edge1.masked_fill_(padding_mask.view(B, 1, L, 1), -torch.inf) # [B, L, L, H]

        ## Shape
        edge0 = edge0.permute(0, 3, 1, 2).contiguous().view(B*H, L, L)
        edge1 = edge1.permute(0, 3, 1, 2).contiguous().view(B*H, L, L)

        # Layers
        for layer0, layer1 in zip(self.layers0, self.layers1):
            (node0, edge0), (node1, edge1) = \
                layer0(node0, edge0, node1, edge1), \
                layer1(node1, edge1, node0, edge0)
        
        # Prediction
        padding_mask = (node != self.node_pad_token).to(torch.float).T # [L, B]
        x = torch.sum(node1*padding_mask.unsqueeze(2), dim=0) / torch.sum(padding_mask, dim=0).unsqueeze(1) # [B, D]
        x = self.predictor(x).view(B) # [B]
        return x

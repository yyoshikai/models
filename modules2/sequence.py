import sys
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..models2 import function_name2func, function_config2func, init_config2func, register_module



# sequence modules
@register_module
class TeacherForcer(nn.Module):
    def __init__(self, length_dim):
        super().__init__()
        self.length_dim = length_dim
        self.input_slices = [slice(None)]*length_dim+[slice(None, -1)]
        self.target_slices = [slice(None)]*length_dim+[slice(1, None)]
    def forward(self, input, return_len=False):
        """
        Parameters
        ----------
        input: (any)[..., legnth, ...]
        return_len(bool):
        
        Returns
        -------
        input: (any)[..., length-1, ...]
        target: [..., length-1, ...]
        """
        return_ = input[self.input_slices], input[self.target_slices]
        if return_len:
            return_ += (return_[-1].shape[self.length_dim], )
        return return_

@register_module
class MaskMaker(nn.Module):
    def __init__(self, mask_token, dtype='bool', direction='equal'):
        super().__init__()
        self.mask_token = mask_token
        self.dtype = dtype
        self.direction = direction
    def forward(self, input: torch.Tensor):
        """
        Parameters
        ----------
        input: (torch.int or long)[...]

        Returns
        -------
        mask: (torch.bool or int)[...]
        """
        if self.direction == 'equal':
            mask = input == self.mask_token
        else:
            mask = input != self.mask_token
        if self.dtype == 'bool':
            pass
        elif self.dtype == 'int':
            mask = mask.to(torch.int)
        return mask

class SelfAttentionLayer_old(nn.TransformerEncoderLayer):
    def __init__(self, d_model, activation, d_ff_factor=None, dim_feedforward=None, norm_first=True, **kwargs):
        """
        Parameters
        """
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(f"Please specify either 'dim_feedforward'({dim_feedforward})"
                +f" XOR 'd_ff_factor'({d_ff_factor})")
        if dim_feedforward is None:
            dim_feedforward = int(d_model*d_ff_factor)
        activation = function_name2func[activation]
        super().__init__(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, norm_first=norm_first, **kwargs)

# 231016 attention weightも返せるようにするため改変 ※__init__の引数の順番が若干変わっている以外は同じ。
@register_module
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, activation, nhead, d_ff_factor=None, 
            dim_feedforward=None, norm_first=True, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(f"Please specify either 'dim_feedforward'({dim_feedforward})"
                +f" XOR 'd_ff_factor'({d_ff_factor})")
        if dim_feedforward is None:
            dim_feedforward = int(d_model*d_ff_factor)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = function_name2func[activation]
    
    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, need_weights=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x1, weight = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, need_weights)
            x = x + x1
            x = x + self._ff_block(self.norm2(x))
        else:
            x1, weight = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + x1)
            x = self.norm2(x + self._ff_block(x))
        if need_weights:
            return x, weight
        else:
            return x
    def _sa_block(self, x, attn_mask, key_padding_mask, need_weights):
        x, weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=need_weights)
        return self.dropout1(x), weights
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

# 重み付きに対応したencoder
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class TransformerEncoderOrg(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights=False):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        output = src
        weights = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                need_weights=need_weights)
            if need_weights:
                output, weight = output
                weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            return output, weights
        else:
            return output

def load_pe_pre_hook_keep(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    state_dict[prefix+"pe"] = model.pe
def load_pe_pre_hook_load(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    if prefix+"pe" in state_dict:
        model.register_buffer('pe', state_dict[prefix+"pe"])
    else:
        state_dict[prefix+"pe"] = model.pe
def load_pe_pre_hook_larger(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    if prefix+"pe" in state_dict and \
        len(model.pe) < len(state_dict[prefix+"pe"]):
        model.register_buffer('pe', state_dict[prefix+"pe"])
    else:
        state_dict[prefix+"pe"] = model.pe
load_pe_pre_hooks = {
    'keep': load_pe_pre_hook_keep,
    'load': load_pe_pre_hook_load,
    'larger': load_pe_pre_hook_larger
}

def get_posenc(length: int, emb_size: int) -> torch.Tensor:
    """
    Returns
    -------
    pe: torch.tensor(float)[length, 1(batch_size dim), emb_size]
    
    """
    pe = torch.zeros(length, emb_size)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_size, 2) *
                            -(math.log(10000.0) / emb_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1)
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_len: int,
            load_pe: str='keep'):
        """
        Only postional encoding part of PositionalEmbedding. See document of
        PositionalEmbedding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = get_posenc(max_len, emb_size)
        self.max_len = max_len
        self.emb_size = emb_size
        self.register_buffer('pe', pe)
        self._register_load_state_dict_pre_hook(load_pe_pre_hooks[load_pe], with_module=True)
    
    def forward(self, input: torch.Tensor, position: int=None):
        """
        Transpose is included.

        Parameters
        ----------
        input: (torch.long)[batch_size, length, embedding_size]
        position(->None): int or None

        Returns
        -------
        output(torch.float)[length, batch_size, embedding_dim]: 
        """
        input = input.transpose(0, 1)
        length = input.shape[0]
        if length > self.max_len:
            print("[WARNING] Overflow in Positional embedding. PE is extended.")
            pe = get_posenc(length=length, emb_size=self.emb_size).to(self.pe.device)
            self.register_buffer('pe', pe)
            self.max_len = length

        if position is None:
            pe = Variable(self.pe[:input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input+pe)

@register_module
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding: dict, dropout: float, max_len:int, 
        factorize:bool=False, load_pe='keep'):
        """
        Parameters
        ----------
        embedding: dict
            Input to nn.Embedding
        dropout: float
            Dropout after positional encoding
        factorize: bool
            True for old Transformer, False for normal Transformer.
        load_pe: str
            How to load model with pe of different length.
        """
        super().__init__()
        self.embedding = nn.Embedding(**embedding)
        self.emb_size = embedding['embedding_dim']
        self.max_len = max_len
        self.factorize = factorize
        if self.factorize:
            self.factor = math.sqrt(self.emb_size)
        self.dropout = nn.Dropout(p=dropout)
        pe = get_posenc(max_len, self.emb_size)
        self.register_buffer('pe', pe)
        self._register_load_state_dict_pre_hook(load_pe_pre_hooks[load_pe], with_module=True)
    
    def forward(self, input: torch.Tensor, position: int=None):
        """
        Transpose is included here.

        Parameters
        ----------
        input: (torch.long)[batch_size, length]
        position(->None): int or None

        Returns
        -------
        output(torch.float)[length, batch_size, embedding_dim]: 
        """
        input = self.embedding(input.transpose(0, 1).contiguous())
        length = input.shape[0]
        if length > self.max_len:
            print("[WARNING] overflow in Positional embedding. PE is extended.")
            pe = get_posenc(length=length, emb_size=self.emb_size).to(self.pe.device)
            self.register_buffer('pe', pe)
            self.max_len = length
        
        if self.factorize:
            input *= self.factor
        if position is None:
            pe = Variable(self.pe[:input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input+pe)

@register_module
class RandomPositionalEncoder(nn.Module):
    def __init__(self, emb_size: int, max_len: int, pad_token: int, factor: float=1.0):
        super().__init__()
        self.emb_size = emb_size
        self.pad_token = pad_token
        self.factor = factor
        pe = get_posenc(length=max_len, emb_size=self.emb_size).squeeze(1) 
        pe *= self.factor
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, input):
        L, D = input.shape
        if L > self.pe.shape[0]:
            print("[WARNING] overflow in RandomPositionalEncoder. PE is extended.")
            pe = get_posenc(L, self.emb_size).squeeze(1).to(self.pe.device) * self.factor
            self.register_buffer('pe', pe, persistent=False)
        position = torch.rand_like(input, dtype=torch.float)
        position.masked_fill_(input == self.pad_token, torch.inf)
        position = torch.argsort(torch.argsort(position, dim=1), dim=1) # それぞれの値が何番目か
        return F.embedding(position, self.pe)
# encoders
@register_module
class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layer, norm=None, init=dict()):
        """
        AttentionEncoderと同じ。

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
        layer = SelfAttentionLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.encoder = TransformerEncoderOrg(layer, num_layers=n_layer, norm=norm)

        # weight init (deprecated)
        init_warned = False
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    if not init_warned:
                        print("[WARNING] Initialization in modules is deprecated. "
                            "Use initialization in model instead.")
                        init_warned = True
                    init_config2func(config)(param)
    
    def forward(self, src, key_padding_mask, need_weights=False):
        """
        Parameters
        ----------
        src: (torch.float)[length, batch_size, d_model]
        key_padding_mask: (torch.float)[batch_size, length]

        Returns
        -------
        memory: (torch.float)[length, batch_size, d_model]
        """
        return self.encoder(src=src, mask=None, src_key_padding_mask=key_padding_mask,
            need_weights=need_weights)

# decoders
@register_module
class TransformerDecoder(nn.Module):
    def __init__(self, layer, n_layer, max_len, norm=None, init=dict()):
        """
        古いモデル。
        (240208では3dvaeにて普通に使っている。)

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
        self.max_len = max_len
        self.register_square_mask(max_len, is_init=True)
        d_model = layer['d_model']
        if 'activation' in layer:
            layer['activation'] = function_config2func(layer['activation'])
        if 'd_ff_factor' in layer:
            layer['dim_feedforward'] = d_model*layer.pop('d_ff_factor')
        layer.setdefault('norm_first', True)
        layer = nn.TransformerDecoderLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer, norm=norm)

        # weight init (deprecated)
        init_warned = False
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    if not init_warned:
                        print("[WARNING] Initialization in modules is deprecated. "
                            "Use initialization in model instead.")
                        init_warned = True
                    init_config2func(config)(param)

    def register_square_mask(self, max_len, is_init=False):
        if not is_init:
            print(f"[WARNING] square mask extended to {max_len}", file=sys.stderr)
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        if not is_init:
            square_mask = square_mask.to(self.square_subsequent_mask.device)
        self.max_len = max_len
        self.register_buffer('square_subsequent_mask', square_mask)

    def forward(self, mode='forced', *args, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Mode to forward
        args, kwargs: 
            See each function for details.
        """
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            raise ValueError(f'Unsupported type of mode: {mode}')
        
    def forced(self, tgt, memory, memory_key_padding_mask):
        """
        Parameters
        ----------
        tgt: (torch.float)[length, batch_size, d_model]
        memory: (torch.float)[length, batch_size, d_model]
        memory_key_padding_mask: (torch.float?)[batch_size, length]

        Returns
        -------
        emb: (torch.float)[batch_size, length, d_model]
        
        """
        length = tgt.shape[0]
        if length > self.max_len:
            self.register_square_mask(length)
        mask = self.square_subsequent_mask[:length, :length]
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=mask, memory_key_padding_mask=memory_key_padding_mask).transpose(0, 1)
    def cell_forward(self, tgt, mem_attn_mask, ks, vs, mem_ks, mem_vs):
        d_model = tgt.shape[-1]
        x = tgt.squeeze(0)
        for i_layer, layer in enumerate(self.decoder.layers):
            residual = x
            attn = layer.self_attn
            num_heads = attn.num_heads
            bsz, embed_dim = x.shape
            head_dim = embed_dim // num_heads
            q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            # q = q.contiguous().view(1, bsz * num_heads, head_dim).transpose(0, 1)
            q = q.contiguous().view(bsz * num_heads, 1, head_dim)
            k = k.contiguous().view(bsz * num_heads, head_dim).unsqueeze(1)
            v = v.contiguous().view(bsz * num_heads, head_dim).unsqueeze(1)
            ks[i_layer] = torch.cat([ks[i_layer], k], dim=1)
            vs[i_layer] = torch.cat([vs[i_layer], v], dim=1)

            dropout_p = attn.dropout if attn.training else 0.0
            attn_output, _ = F._scaled_dot_product_attention(q, ks[i_layer], vs[i_layer], None, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, embed_dim)
            attn_output = attn.out_proj(attn_output)
            attn_output = attn_output.view(bsz, attn_output.size(1))
            x = attn_output
            x = layer.norm1(layer.dropout1(x)+residual)

            residual = x
            attn = layer.multihead_attn
            num_heads = attn.num_heads
            bsz, embed_dim = x.shape
            head_dim = embed_dim // num_heads
            q =  F.linear(x, attn.in_proj_weight[:d_model], attn.in_proj_bias[:d_model])
            q = q.contiguous().view(1, bsz * num_heads, head_dim).transpose(0, 1)
            dropout_p = 0.0 if not attn.training else attn.dropout
            attn_output, _ = F._scaled_dot_product_attention(q, mem_ks[i_layer], mem_vs[i_layer], mem_attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, embed_dim)
            attn_output = attn.out_proj(attn_output)
            attn_output = attn_output.view(bsz, attn_output.size(1))
            x = attn_output
            x = layer.norm2(layer.dropout2(x)+residual)
            x = layer.norm3(layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))+x)
        out = self.decoder.norm(x)
        return out.unsqueeze(1)

    def prepare_cell_forward(self, memory, memory_key_padding_mask):
        ilen, bsize, d_model = memory.shape
        nhead = self.decoder.layers[0].multihead_attn.num_heads
        n_layer = len(self.decoder.layers)
        device = memory.device
        mem_attn_mask = torch.zeros_like(memory_key_padding_mask, dtype=memory.dtype)
        mem_attn_mask.masked_fill_(memory_key_padding_mask, float("-inf"))
        mem_attn_mask = mem_attn_mask.view(bsize, 1, 1, ilen). \
            expand(-1, nhead, -1, -1).reshape(bsize * nhead, 1, ilen)

        ks = [torch.full((bsize*nhead, 0, d_model//nhead), fill_value=0.0, device=device) for i in range(n_layer)]
        vs = [torch.full((bsize*nhead, 0, d_model//nhead), fill_value=0.0, device=device) for i in range(n_layer)]
        mem_ks = []
        mem_vs = []
        for layer in self.decoder.layers:
            attn = layer.multihead_attn
            w_kv = attn.in_proj_weight[d_model:]
            b_kv = attn.in_proj_bias[d_model:]
            
            kv =  F.linear(memory, w_kv, b_kv)
            k, v = kv.chunk(2, dim=-1)
            k = k.contiguous().view(k.shape[0], bsize * nhead, d_model//nhead).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], bsize * nhead, d_model//nhead).transpose(0, 1)
            mem_ks.append(k)
            mem_vs.append(v)
        return mem_attn_mask, ks, vs, mem_ks, mem_vs
    def expand_beam(self, memory: torch.Tensor, 
            memory_key_padding_mask: torch.Tensor, beam_size: int):
        """
        Parameters
        ----------
        memory: (torch.float)[length, batch_size, d_model]
        memory_key_padding_mask: (torch.bool?float?)
            [batch_size, length]
        """
        length, batch_size, d_model = memory.shape
        memory = memory.view(length, batch_size, 1, d_model) \
            .expand(-1, -1, beam_size, -1) \
            .contiguous() \
            .view(length, batch_size*beam_size, d_model)
        memory_key_padding_mask = memory_key_padding_mask \
            .view(batch_size, 1, length) \
            .expand(-1, beam_size, -1) \
            .contiguous() \
            .view(batch_size*beam_size, length)
        return memory, memory_key_padding_mask
    def gather_beam(self, beam_index: torch.Tensor, 
            ks, vs, mem_ks, mem_vs):
        """
        Parameters
        ----------
        beam_index: (long)[batch_size, beam_size]
        
        
        """
        batch_size, beam_size = beam_index.shape
        batch_beam_head_size, dlength, d_head = ks[0].shape
        head_size = batch_beam_head_size // batch_size // beam_size
        beam_index_d = beam_index \
            .view(batch_size, beam_size, 1, 1, 1) \
            .expand(-1, -1, head_size, dlength, d_head)
        new_ks = [
            k.view(batch_size, beam_size, head_size, dlength, d_head) \
                .gather(dim=1, index=beam_index_d) \
                .view(batch_size*beam_size*head_size, dlength, d_head)
            for k in ks]
        new_vs = [
            v.view(batch_size, beam_size, head_size, dlength, d_head) \
                .gather(dim=1, index=beam_index_d) \
                .view(batch_size*beam_size*head_size, dlength, d_head)
            for v in vs]
        
        _, mlength, d_model = mem_ks[0].shape
        beam_index_m = beam_index \
            .view(batch_size, beam_size, 1, 1, 1) \
            .expand(batch_size, beam_size, head_size, mlength, d_model)
        new_mem_ks = [
            mem_k.view(batch_size, beam_size, head_size, mlength, d_model) \
                .gather(dim=1, index=beam_index_m) \
                .view(batch_size*beam_size*head_size, mlength, d_model)
            for mem_k in mem_ks]
        new_mem_vs = [
            mem_v.view(batch_size, beam_size, head_size, mlength, d_model) \
                .gather(dim=1, index=beam_index_m) \
                .view(batch_size*beam_size*head_size, mlength, d_model)
            for mem_v in mem_vs]

        return new_ks, new_vs, new_mem_ks, new_mem_vs

def load_square_mask_pre_hook_keep(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
def load_square_mask_pre_hook_load(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    if prefix+"square_subsequent_mask" in state_dict:
        model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
    else:
        state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
def load_square_mask_pre_hook_larger(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    if prefix+"square_subsequent_mask" in state_dict and \
        len(model.square_subsequent_mask) < len(state_dict[prefix+"square_subsequent_mask"]):
        model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
    else:
        state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
load_square_mask_pre_hooks = {
    'keep': load_square_mask_pre_hook_keep,
    'load': load_square_mask_pre_hook_load,
    'larger': load_square_mask_pre_hook_larger,
}


"""
このクラスは最小限の機能のみで, あくまで各クラスを優先する
"""
class LatentSequenceDecoder(nn.Module):
    def forward(self, mode='forced', *args, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Mode to forward
        args, kwargs: 
            See each function for details.
        """
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            raise ValueError(f'Unsupported type of mode: {mode}')
    def forced(self, *args, **kwargs):
        raise NotImplementedError
    def cell_forward(self, *args, **kwargs):
        raise NotImplementedError
    def prepare_cell_forward(self, *args, **kwargs):
        raise NotImplementedError
    def split_beam(self, *args, **kwargs):
        raise NotImplementedError

    def register_square_mask(self, max_len, is_init=False):
        if not is_init:
            print(f"[WARNING] square mask extended to {max_len}", file=sys.stderr)
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        if not is_init:
            square_mask = square_mask.to(self.square_subsequent_mask.device)
        self.max_len = max_len
        self.register_buffer('square_subsequent_mask', square_mask)

@register_module
class AttentionDecoder(LatentSequenceDecoder):
    def __init__(self, layer, num_layers, max_len, init={}, load_square_mask='keep'):
        """
        layer: dict
            input for SelfAttentionLayer
        num_layers: int
        init: dict
            Initialization for each parameter
        max_len: int
        load_square_mask: いる?
        """
        super().__init__()
        self.max_len = max_len
        self.register_square_mask(max_len, is_init=True)
        d_model = layer['d_model']
        self.d_model = d_model

        # decoder
        decoder_layer = SelfAttentionLayer_old(**layer)
        self.decoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)
        
        # weight init (deprecated)
        init_warned = False
        for layer in self.decoder.layers:
            for param_name in init:
                if not init_warned:
                    print("[WARNING] Initialization in modules is deprecated. "
                        "Use initialization in model instead.")
                    init_warned = True
                init_config2func(init[param_name])(layer.state_dict()[param_name]) 

        # define operation in load_state_dict
        self._register_load_state_dict_pre_hook(load_square_mask_pre_hooks[load_square_mask],
            with_module=True)

    def prepare_cell_forward(self, latent: torch.Tensor):
        """
        Parameters
        ----------
        latent: torch.tensor(float)[batch_size, d_model]

        Returns
        -------
        state: [torch.tensor(float)[0, batch_size, d_model]]
        
        """
        batch_size, d_model = latent.shape
        return [torch.zeros(size=(0, batch_size, self.d_model), dtype=torch.float, device=latent.device)
            for i_layer in range(self.decoder.num_layers)]
    def gather_beam(self, state, beam_index: torch.Tensor):
        """
        Parameters
        ----------
        state: list[(float)[length, batch_size*beam_size, d_model]]
        beam_index: (long)[batch_size, beam_size]
        """
        length, _, d_model = state[0].shape
        batch_size, beam_size = beam_index.shape
        new_state = []
        beam_index = beam_index.view(1, batch_size, beam_size, 1).expand(length, -1, -1, d_model)
        for state0 in state:
            state0 = state0.view(length, batch_size, beam_size, d_model)
            state0 = state0.gather(dim=2, index=beam_index)
            state0 = state0.view(length, -1, d_model)
            new_state.append(state0)
        return new_state

    # tgt, memory, memory_key_padding_mask
    def forced(self, tgt, latent):
        """
        Parameters
        ----------
        tgt: (float)[max_len, batch_size, d_model]
        latent (float)[batch_size, d_model]

        Returns
        -------
        output: (float)[batch_size, max_len, d_model]
        """
        max_len, _, _ = tgt.shape
        if max_len > self.max_len:
            self.register_square_mask(max_len)
        tgt = tgt + latent.unsqueeze(0)
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=tgt, mask=input_mask, src_key_padding_mask=None)
        return output.transpose(0, 1)

    def cell_forward(self, tgt, latent, state, position):
        """
        Parameters
        ----------
        tgt: (float)[1, batch_size, d_model]
            embedded input at (position) th place.
        latent: (float)[batch_size, d_model]
            latent representation

        Returns
        -------
        cur_output(float)[batch_size, 1, d_model]:
            Output of decoder
        state: [(float)[length, batch_size, d_model])]
        """

        cur_output = (tgt + latent.unsqueeze(0))
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(cur_y, y, y, attn_mask=None,
                        key_padding_mask=None, need_weights=False)[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        
        return cur_output.transpose(0, 1), state

    def expand_beam(self, latent: torch.Tensor, beam_size: int):
        """
        Parameters
        ----------
        latent: (float)[length, batch_size, d_model]
        
        """
        length, batch_size, d_model = latent.shape
        latent = latent.view(length, batch_size, 1, d_model)
        latent = latent.expand(-1, batch_size, -1, -1).contiguous()
        latent = latent.view(length, batch_size*beam_size, d_model)
        return latent


# LMベースのmemoryやlatentを必要としないDecoder
@register_module
class TransformerLMDecoder(LatentSequenceDecoder):
    def __init__(self, layer, num_layers, init, max_len, load_square_mask='keep'):
        """
        layer: dict
            input for SelfAttentionLayer
        num_layers: int
        init: dict
            Initialization for each parameter
        max_len: int
        load_square_mask: いる?
        """
        super().__init__()
        self.max_len = max_len
        self.register_square_mask(max_len, is_init=True)
        d_model = layer['d_model']
        self.d_model = d_model

        # decoder
        decoder_layer = SelfAttentionLayer_old(**layer)
        self.decoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)

        # weight init (deprecated)
        init_warned = False
        for layer in self.decoder.layers:
            for param_name in init:
                if not init_warned:
                    print("[WARNING] Initialization in modules is deprecated. "
                        "Use initialization in model instead.")
                    init_warned = True
                init_config2func(init[param_name])(layer.state_dict()[param_name]) 

        # define operation in load_state_dict
        self._register_load_state_dict_pre_hook(load_square_mask_pre_hooks[load_square_mask], with_module=True)
    
    def prepare_cell_forward(self, batch_size):
        """
        Parameters
        ----------
        batch_size: int

        Returns
        -------
        state: [torch.tensor(float)[0, batch_size, d_model]]
        
        """
        return [torch.zeros(size=(0, batch_size, self.d_model), dtype=torch.float, device=self.square_subsequent_mask.device)
            for i_layer in range(self.decoder.num_layers)]

    def forced(self, tgt):
        """
        Parameters
        ----------
        tgt: (float)[max_len, batch_size, d_model]
        latent (float)[batch_size, d_model]

        Returns
        -------
        output: (float)[batch_size, max_len, d_model]
        """
        max_len, _, _ = tgt.shape
        if max_len > self.max_len:
            self.register_square_mask(max_len)
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=tgt, mask=input_mask, src_key_padding_mask=None)
        return output.transpose(0, 1)

    def cell_forward(self, tgt, state, position):
        """
        Parameters
        ----------
        tgt: (float)[1, batch_size, d_model]
            embedded input at (position) th place.
        latent: (float)[batch_size, d_model]
            latent representation

        Returns
        -------
        cur_output(float)[batch_size, 1, d_model]:
            Output of decoder
        state: [(float)[length, batch_size, d_model])]
        """

        cur_output = tgt
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(cur_y, y, y, attn_mask=None,
                        key_padding_mask=None, need_weights=False)[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        
        return cur_output.transpose(0, 1), state


@register_module
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n_class = input.shape[-1]
        return super().forward(input=input.contiguous().view(-1, n_class), target=target.ravel())

@register_module
class GreedyDecoder(nn.Module):
    """
    231013 クラス自体にデータを貯めるようになっていないのはなぜですか?    
    """
    def __init__(self, start_token, end_token = None):
        super().__init__()
        self.start_token = start_token
        if end_token is None:
            print("[WARNING] end_token is not specified in GreedyDecoder.__init__ and defaulted to 2", 
                  file=sys.stderr)
            end_token = 2
        self.end_token = end_token
        self._device_param = nn.Parameter(torch.zeros((0,)))
    def forward(self, *args, mode, **kwargs):
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported type of mode: {mode}")
    def init(self, batch_size):
        cur_input = torch.full((batch_size, 1), fill_value=self.start_token,
            dtype=torch.long, device=self._device_param.device)
        self.is_decoded = torch.full((batch_size, ), fill_value=False, 
            dtype=torch.bool, device=self._device_param.device)
        return cur_input, []
    def add(self, cur_proba, outs, return_end=False):
        """
        cur_proba: (float)[batch_size, 1, n_token]
        outs: list[torch.tensor[batch_size, 1]]
        """
        cur_input = torch.argmax(cur_proba, dim=-1)
        outs.append(cur_input)
        output = cur_input, outs
        if return_end:
            self.is_decoded = torch.logical_or(self.is_decoded, cur_input == self.end_token)
            if torch.all(self.is_decoded):
                output += (True, )
            else:
                output += (False, )

        return output

    def sample_add(self, cur_proba, outs):
        """
        cur_proba: (float)[batch_size, 1, n_token]
        outs: list
        """
        cur_input = torch.multinomial(F.softmax(cur_proba.squeeze(1), dim=-1), num_samples=1)
        outs.append(cur_input)
        return cur_input, outs
    
    def aggregate(self, outs):
        return torch.cat(outs, dim=1)
    
    def beam_init(self, latent: torch.Tensor, beam_size: int, expand: bool=True):
        """
        Parameters
        ----------
        latent: (float)[batch_size, latent_size]
        beam_size: int

        Returns
        -------
        latent: (float)[batch_size*beam_size, latent_size]
        cur_input: (float)[batch_size*beam_size, 1]
        is_ended: (bool)[batch_size, beam_size]
        outs: (long)[0(length), batch_size, beam_size]
        proba: (float)[batch_size, beam_size]
        
        """
        batch_size = latent.shape[-2]
        device = latent.device
        if expand: 
            latent_size = latent.shape[1]
            latent = latent.view(batch_size, 1, latent_size).expand(-1, beam_size, -1).contiguous()
            latent = latent.view(batch_size*beam_size, latent_size)
        cur_input = torch.full((batch_size*beam_size, 1), fill_value=self.start_token,
            dtype=torch.long, device=device)
        is_ended = torch.full((batch_size, beam_size), fill_value=False,
            dtype=torch.bool, device=device)
        outs = torch.zeros((0, batch_size, beam_size), dtype=torch.long, device=device)
        proba = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        return latent, cur_input, outs, proba, is_ended
    def beam_add(self, cur_proba: torch.Tensor, proba: torch.Tensor, 
            outs: torch.Tensor, is_ended: torch.Tensor):
        """
        Parameters
        ----------
        cur_proba: (float)[batch_size*beam_size, 1, voc_size]
        proba: (float)[batch_size, beam_size]
            ※softmax前
        outs: (long)[length, batch_size, beam_size]
        is_ended: (bool)[batch_size, beam_size]
        
        B: batch_size
        E: beam_size
        V: voc_size
        """
        _, _, voc_size = cur_proba.shape
        length, batch_size, beam_size = outs.shape
        cur_proba = cur_proba.view(batch_size, beam_size, voc_size)

        # mask ended probas
        cur_proba[is_ended] = -torch.inf
        cur_proba[:,:,self.end_token][is_ended] = 0
        proba = proba.unsqueeze(-1) + cur_proba
        proba = proba.view(batch_size, -1) # [B, E*V]
        proba, topk_beam_voc = proba.topk(k=beam_size, dim=-1) # [B, E]
        topk_voc = topk_beam_voc % voc_size # [B, E]
        topk_beam = torch.div(topk_beam_voc, voc_size, rounding_mode='floor') # [B, E]

        # gather values
        outs = outs.gather(dim=-1, index=topk_beam.view(1, batch_size, beam_size) \
            .expand((length, batch_size, beam_size)))
        is_ended = is_ended.gather(dim=-1, index=topk_beam)

        outs = torch.cat([
            outs,
            topk_voc.unsqueeze(0)
        ], dim=0)
        is_ended[topk_voc == self.end_token] = True
        cur_input = topk_voc.view(batch_size*beam_size, 1)
        return cur_input, proba, outs, is_ended, topk_beam
    def beam_aggregate(self, outs: torch.Tensor):
        """
        Parameters
        ----------
        outs: (long)[length, batch_size, beam_size]

        Returns
        -------
        outs: (long)[batch_size, length]
        """
        return outs[:,:,0].transpose(0, 1).contiguous()

    def force(self, proba, add_start_token=False):
        force = torch.argmax(proba, dim=-1)
        if add_start_token:
            batch_size, length = force.shape
            force = torch.cat([torch.full((batch_size, 1), fill_value=self.start_token),
                force])
        return force

def get_token_size(input: torch.Tensor, pad_token: int):
    return torch.sum(input != pad_token)
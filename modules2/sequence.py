import math
from functools import partial
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..models2 import function_name2func, function_config2func, \
    init_config2func, module_type2class
import re

# sequence modules
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

class SelfAttentionLayer(nn.TransformerEncoderLayer):
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

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding: dict, dropout: float, max_len:int, 
        factorize:bool=False):
        """
        Parameters
        ----------
        embedding: dict
            Input to nn.Embedding
        dropout: float
            Dropout after positional encoding
        factorize: bool
            True for old Transformer, False for normal Transformer.
        """
        super().__init__()
        self.embedding = nn.Embedding(**embedding)
        emb_size = embedding['embedding_dim']
        self.factorize = factorize
        if self.factorize:
            self.factor = math.sqrt(emb_size)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) *
                             -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, input, position: int=None):
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
        if self.factorize:
            input *= self.factor
        if position is None:
            pe = Variable(self.pe[:input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input+pe)

# encoders
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, norm=norm)

        # weight init
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name: # re.fullmatchで合わせたいが, ピリオドが特殊文字なので保留中
                    init_config2func(config)(param)
    
    def forward(self, src, key_padding_mask):
        """
        Parameters
        ----------
        src: (torch.float)[length, batch_size, d_model]
        key_padding_mask: (torch.float)[batch_size, length]

        Returns
        -------
        memory: (torch.float)[length, batch_size, d_model]
        """
        return self.encoder(src=src, mask=None, src_key_padding_mask=key_padding_mask)

# decoders
class TransformerDecoder(nn.Module):
    def __init__(self, layer, n_layer, max_len, norm=None, init=dict()):
        """
        古いモデル。

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
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('square_subsequent_mask', square_mask)
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

        # weight init
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name: # re.fullmatchで合わせたいが, ピリオドが特殊文字なので保留中
                    init_config2func(config)(param)

    def forward(self, mode='forced', *args, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Mode to forward
        args, kwargs: 
            See each function for details.
        """
        if mode == 'forced':
            return self.forced(*args, **kwargs)
        elif mode == 'cell_forward':
            return self.cell_forward(*args, **kwargs)
        elif mode == 'prepare_cell_forward':
            return self.prepare_cell_forward(*args, **kwargs)
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
        if mode == 'forced':
            return self.forced(*args, **kwargs)
        elif mode == 'cell_forward':
            return self.cell_forward(*args, **kwargs)
        elif mode == 'prepare_cell_forward':
            return self.prepare_cell_forward(*args, **kwargs)
        else:
            raise ValueError(f'Unsupported type of mode: {mode}')
    def forced(self, *args, **kwargs):
        raise NotImplementedError
    def cell_forward(self, *args, **kwargs):
        raise NotImplementedError
    def prepare_cell_forward(self, *args, **kwargs):
        raise NotImplementedError

class AttentionDecoder(LatentSequenceDecoder):
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
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('square_subsequent_mask', square_mask)
        d_model = layer['d_model']
        self.d_model = d_model

        # decoder
        decoder_layer = SelfAttentionLayer(**layer)
        self.decoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)
        
        # weight init
        for layer in self.decoder.layers:
            for param_name in init:
                init_config2func(init[param_name])(layer.state_dict()[param_name]) 

        # define operation in load_state_dict
        if load_square_mask == 'keep':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'load':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict:
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'larger':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict and \
                    len(model.square_subsequent_mask) < len(state_dict[prefix+"square_subsequent_mask"]):
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        else:
            raise ValueError(f"Unsupported type of config.load_square_mask: {load_square_mask}")
        self._register_load_state_dict_pre_hook(pre_hook, with_module=True)
 
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

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n_class = input.shape[-1]
        return super().forward(input=input.contiguous().view(-1, n_class), target=target.ravel())

class GreedyDecoder(nn.Module):
    def __init__(self, start_token):
        super().__init__()
        self.start_token = start_token
        self._device_param = nn.Parameter(torch.zeros((0,)))
    def forward(self, *args, mode, **kwargs):
        if mode == 'init':
            return self.init(*args, **kwargs)
        if mode == 'add':
            return self.add(*args, **kwargs)
        elif mode == 'aggregate':
            return self.aggregate(*args, **kwargs)
        elif mode == 'force':
            return self.force(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported type of mode: {mode}")
    def init(self, batch_size):
        cur_input = torch.full((batch_size, 1), fill_value=self.start_token,
            dtype=torch.long, device=self._device_param.device)
        return cur_input, []
    def add(self, cur_proba, outs):
        cur_input = torch.argmax(cur_proba, dim=-1)
        outs.append(cur_input)
        return cur_input, outs
    def aggregate(self, outs):
        return torch.cat(outs, dim=1)
    def force(self, proba, add_start_token=False):
        force = torch.argmax(proba, dim=-1)
        if add_start_token:
            batch_size, length = force.shape
            force = torch.cat([torch.full((batch_size, 1), fill_value=self.start_token),
                force])
        return force

class BeamSearcher(nn.Module):
    def __init__(self, start_token):
        super().__init__()
        raise NotImplementedError

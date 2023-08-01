import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..models2 import function_name2func, Model


# sequence modules
class TeacherForcer(nn.Module):
    def __init__(self, length_dim):
        super().__init__()
        self.input_slices = [slice(None)]*length_dim+[slice(None, -1)]
        self.target_slices = [slice(None)]*length_dim+[slice(1, None)]
    def forward(self, input):
        """
        input: torch.tensor(any)[..., legnth, ...]
        output:
          - torch.tensor(any)[..., length-1, ...]
          - torch.tensor(any)[..., length-1, ...]
        """
        return input[self.input_slices], input[self.target_slices]

class MaskMaker(nn.Module):
    def __init__(self, mask_token, type='bool', direction='equal'):
        super().__init__()
        self.mask_token = mask_token
        self.type_ = type
        self.direction = direction
    def forward(self, input):
        if self.direction == 'equal':
            mask = input == self.mask_token
        else:
            mask = input != self.mask_token
        if self.type_ == 'bool':
            pass
        return mask

class SelfAttentionLayer(nn.Module):
    def __init__(self, mha, layernorm, dropout, activation, 
            dim_feedforward=None, d_ff_factor=None):
        """
        Parameters
        ----------
        mha: dict
            Argument for torch.nn.MultiheadAttention
            d_model: int
                Dimension of model
        layernorm: dict
            Argument for torch.nn.MultiheadAttention
        dropout: float
        activation: str
        dim_feedforward: int[Optional]
        d_ff_factor: float[Optional]
        """
        super().__init__()
        d_model = mha['d_model']
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(f"Please specify either 'dim_feedforward'({dim_feedforward})"
                +f" XOR 'd_ff_factor'({d_ff_factor})")
        if dim_feedforward is None:
            dim_feedforward = int(d_model*d_ff_factor)
        self.norm1 = nn.LayerNorm(d_model, **layernorm)
        self.self_attn = nn.MultiheadAttention(**mha)
        self.norm2 = nn.LayerNorm(d_model, **layernorm)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = function_name2func[activation]
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        cur_input = src
        cur_input = cur_input + self._sa_block(self.norm1(cur_input), src_mask, src_key_padding_mask)
        cur_input = cur_input + self._ff_block(self.norm2(cur_input))
        return cur_input
    def _sa_block(self, cur_input, attn_mask, key_padding_mask):
        cur_input = self.self_attn(cur_input, cur_input, cur_input,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(cur_input)
    def _ff_block(self, cur_input):
        cur_input = self.linear2(self.dropout(self.activation(self.linear1(cur_input))))
        return self.dropout2(cur_input)

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding: dict, dropout: float, max_len:int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(**embedding)
        emb_size = embedding['embedding_dim']
        self.factor = math.sqrt(emb_size)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) *
                             -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, input):
        """
        input: torch.tensor(torch.long)[batch_size, length]
        
        """
        input = self.embedding(input) * self.factor
        input = input + Variable(self.pe[:, :input.size(1)], requires_grad=False)
        return self.dropout(input)


# functions
def NewGELU(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) \
        * (input + 0.044715 * torch.pow(input, 3.0))))
def sigmoid(cur_input):
    return 1/(1+math.e**(-cur_input))

def init_config2func(layer_config):
    if type(layer_config) == str:
        name = layer_config
    elif type(layer_config) in {int, float}:
        name = layer_config
    elif layer_config == {}:
        name = 'none'
    else:
        name = layer_config.type
    if type(name) in {int, float}:
        return lambda cur_input: nn.init.constant_(cur_input, float(name))
    if name == 'glorot_uniform':
        return nn.init.xavier_uniform_
    elif name == 'glorot_normal':
        return nn.init.xavier_normal_
    elif name == 'he_uniform':
        return nn.init.kaiming_uniform_
    elif name == 'he_normal':
        return nn.init.kaiming_normal_
    elif name == 'uniform':
        return lambda cur_input: nn.init.uniform_(cur_input, layer_config.a, layer_config.b)
    elif name == 'normal':
        return lambda cur_input: nn.init.normal_(cur_input, layer_config.mean, layer_config.std)
    elif name in ['zero', 'zeros']:
        return nn.init.zeros_
    elif name in ['one', 'ones']:
        return nn.init.ones_
    elif name == 'none':
        return lambda cur_input: None
    else:
        raise ValueError(f"Unsupported types of init function: {layer_config}")

function_name2func = {
    'relu': F.relu,
    'gelu': F.gelu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'newgelu': NewGELU,
    'none': lambda cur_input: cur_input,
    'exp': torch.exp,
    'log': torch.log,
    'sum': torch.sum,
    'mean': torch.mean,
    'log_softmax': F.log_softmax,
    'softplus': F.softplus
}
def function_config2func(config):
    if config.type in function_name2func:
        args = config.copy()
        type_ = args.pop('type')
        if len(args) == 0:
            return function_name2func[config.type]
        else:
            return partial(function_name2func[type_], **args)
    elif config.type == 'affine':
        weight = config.weight if config.weight else 1.0
        bias = config.bias if config.bias else 0.0
        return lambda cur_input: cur_input*weight+bias
    else:
        raise ValueError(f"Unsupported type of function config: {config.type}")

# encoder decoders
class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layer, norm=None):
        super().__init__()
        # process layer config
        d_model = layer['d_model']
        if 'activation' in layer:
            layer['activation'] = function_config2func(layer['activation'])
        if 'd_ff_factor' in layer:
            layer['dim_feedforward'] = d_model*layer.pop('d_ff_factor')
        layer = nn.TransformerEncoderLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, norm=norm)
    def forward(self, src, key_padding_mask):
        """
        src: torch.tensor(float)[batch_size, length, d_model]
        key_padding_mask: torch.tensor(float)[batch_size, length]
        """
        src = src.transpose(0, 1).contiguous()
        return self.encoder(src=src, mask=None, src_key_padding_mask=key_padding_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, layer, n_layer, max_len, norm=None):
        """
        Parameters
        ----------
        layer: arguments for 
        
        
        """
        super().__init__()
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('square_subsequent_mask', square_mask)
        d_model = layer['d_model']
        if 'activation' in layer:
            layer['activation'] = function_config2func(layer['activation'])
        if 'd_ff_factor' in layer:
            layer['dim_feedforward'] = d_model*layer.pop('d_ff_factor')
        layer = nn.TransformerDecoderLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer, norm=norm)

    def forward(self, mode='forced', *args, **kwargs):
        if mode == 'forced':
            return self.forced(*args, **kwargs)
        elif mode == 'cell_forward':
            return self.cell_forward(*args, **kwargs)
        elif mode == 'prepare_greedy':
            return self.prepare_greedy(*args, **kwargs)
    def forced(self, memory, memory_key_padding_mask):
        length = tgt.shape[1]
        mask = self.square_subsequent_mask[:length, :length]
        tgt = tgt.transpose(0, 1).contiguous()
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=mask, memory_key_padding_mask=memory_key_padding_mask).transpose(0, 1)
    def prepare_greedy(self, memory, memory_key_padding_mask, start_token):
        batch_size, src_len, d_model = memory.shape[0]
        nhead = self.decoder.layers[0].multihead_attn.num_heads
        n_layer = len(self.decoder.layers)
        device = memory.device

        mem_attn_mask = torch.zeros_like(memory_key_padding_mask, dtype=torch.float)
        mem_attn_mask.masked_fill_(memory_key_padding_mask, float("-inf"))
        mem_attn_mask = mem_attn_mask.view(batch_size, 1, 1, src_len).   \
                    expand(-1, nhead, -1, -1).reshape(batch_size*nhead, 1, src_len)

        cur_input = torch.full_like((1, batch_size), fill_value=start_token, 
            dtype=torch.long, device=device)
        # calcluate memory
        ks = [torch.full((batch_size*nhead, 0, d_model//nhead), fill_value=0.0) for i in range(n_layer)]
        vs = [torch.full((batch_size*nhead, 0, d_model//nhead), fill_value=0.0) for i in range(n_layer)]
        mem_ks = []
        mem_vs = []
        for layer in self.decoder.layers:
            attn = layer.multihead_attn
            kv =  F.linear(memory, attn.in_proj_weight[d_model:], attn.in_proj_bias[d_model:])
            k, v = kv.chunk(2, dim=-1)
            k = k.contiguous().view(k.shape[0], batch_size * nhead, d_model//nhead).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], batch_size * nhead, d_model//nhead).transpose(0, 1)
            mem_ks.append(k)
            mem_vs.append(v)
        return cur_input, mem_attn_mask, ks, vs, mem_ks, mem_vs
    def cell_forward(self, cur_input, mem_attn_mask, ks, vs, mem_ks, mem_vs):
        batch_size, _, d_model = cur_input.shape

        for i_layer, layer in enumerate(self.decoder.layers):
            residual = cur_input
            attn = layer.self_attn
            num_heads = attn.num_heads
            head_dim = d_model // num_heads
            q, k, v = F.linear(cur_input, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            q = q.contiguous().view(batch_size * num_heads, head_dim).unsqueeze(1)
            k = k.contiguous().view(batch_size * num_heads, head_dim).unsqueeze(1)
            v = v.contiguous().view(batch_size * num_heads, head_dim).unsqueeze(1)
            ks[i_layer] = torch.cat([ks[i_layer], k], dim=1)
            vs[i_layer] = torch.cat([vs[i_layer], v], dim=1)

            dropout_p = attn.dropout if attn.training else 0.0
            attn_output, _ = F._scaled_dot_product_attention(q, ks[i_layer], vs[i_layer], None, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(1 * batch_size, d_model)
            attn_output = attn.out_proj(attn_output)
            attn_output = attn_output.view(1, batch_size, attn_output.size(1))
            cur_input = attn_output
            cur_input = layer.norm1(layer.dropout1(cur_input)+residual)

            residual = cur_input
            attn = layer.multihead_attn
            num_heads = attn.num_heads
            head_dim = d_model // num_heads
            q =  F.linear(cur_input, attn.in_proj_weight[:d_model], attn.in_proj_bias[:d_model])
            q = q.contiguous().view(batch_size * num_heads, head_dim).unsqueeze(1)
            dropout_p = 0.0 if not attn.training else attn.dropout
            attn_output, _ = F._scaled_dot_product_attention(q, mem_ks[i_layer], mem_vs[i_layer], mem_attn_mask, dropout_p)
            attn_output = attn_output.transpose(0, 1).contiguous().view(1 * batch_size, d_model)
            attn_output = attn.out_proj(attn_output)
            attn_output = attn_output.view(1, batch_size, attn_output.size(1))
            cur_input = attn_output
            cur_input = layer.norm2(layer.dropout2(cur_input)+residual)
            cur_input = layer.norm3(layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(cur_input)))))+cur_input)
        if self.decoder.norm is not None:
            cur_input = self.decoder.norm(cur_input)
        return cur_input, ks, vs, mem_ks, mem_vs
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n_class = input.shape[-1]
        return super().forward(input=input.view(-1, n_class), target=target.ravel())
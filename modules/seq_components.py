
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

EMPTY = lambda x: x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, load_pe='keep'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # [position, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

        # define operation in load_state_dict
        if load_pe == 'keep':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                state_dict[prefix+"pe"] = model.pe
        elif load_pe == 'load':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"pe" in state_dict:
                    model.register_buffer('pe', state_dict[prefix+"pe"])
                else:
                    state_dict[prefix+"pe"] = model.pe
        elif load_pe == 'larger':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"pe" in state_dict and len(model.pe) < len(state_dict[prefix+"pe"]):
                    model.register_buffer('pe', state_dict[prefix+"pe"])
                else:
                    state_dict[prefix+"pe"] = model.pe
        else:
            raise ValueError(f"Unsupported type of load_pe: {load_pe}")
        self._register_load_state_dict_pre_hook(pre_hook, with_module=True)                

    def forward(self, x):
        """
        x: torch.tensor of torch.float [max_len, batch_size, d_model]
        """
        x = x + Variable(self.pe[:x.size(0)])
        return self.dropout(x)
    def encode_position(self, x, position):
        return self.dropout(x + self.pe[position])

# Poolers
class MeanPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
    def forward(self, input_, padding_mask):
        """
        Parameters
        ----------
        input: torch.tensor of torch.float [length, batch_size, hidden_size]
        padding_mask: torch.tensor of torch.long [batch_size, length]
            1: 
        """
        return torch.sum(input_*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1)

class StartPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
    def forward(self, input_, padding_mask):
        return input_[0]

class MeanStartPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*2
    def forward(self, input_, padding_mask):
        return torch.cat([torch.sum(input_*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1),
        input_[0]], dim=-1)
        
class MaxPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
    def forward(self, input_, padding_mask):
        input_ = input_ + torch.log(padding_mask).unsqueeze(-1)
        return torch.max(input_, dim=0)[0] 

class MeanStartMaxPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.poolers = nn.ModuleList(
            [MeanPooler(input_size), StartPooler(input_size), MaxPooler(input_size)])
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input_, padding_mask):
        masked_input = input_ + torch.log(padding_mask).unsqueeze(-1)
        return torch.cat([
            torch.sum(input_*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1),
            input_[0],
            torch.max(masked_input, dim=0)[0]], dim=-1)

class MeanStartMaxNormPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_size[-1])
        self.start_norm = nn.LayerNorm(input_size[-1])
        self.max_norm = nn.LayerNorm(input_size[-1])
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input_, padding_mask):
        masked_input = input_ + torch.log(padding_mask).unsqueeze(-1)
        return torch.cat([
            self.mean_norm(torch.sum(input_*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1)),
            self.start_norm(input_[0]),
            self.max_norm(torch.max(masked_input, dim=0)[0])], dim=-1)

class NoAffinePooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.start_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.max_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input_, padding_mask):
        masked_input = input_ + torch.log(padding_mask).unsqueeze(-1)
        return torch.cat([
            self.mean_norm(torch.sum(input_*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1)),
            self.start_norm(input_[0]),
            self.max_norm(torch.max(masked_input, dim=0)[0])], dim=-1)

class NemotoPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input_, padding_mask):
        mx = torch.max(input_,dim=0)[0]
        ave = torch.mean(input_,dim=0)
        first = input_[0]
        return torch.cat([mx,ave,first],dim=1)

pooler_type2class = {
    'start': StartPooler,
    'mean': MeanPooler,
    'meanstart': MeanStartPooler,
    'startmean': MeanStartPooler,
    'meanstartmax': MeanStartMaxPooler,
    'meanmaxstart': MeanStartMaxPooler,
    'norm': MeanStartMaxNormPooler,
    'meanstartmaxnorm': MeanStartMaxNormPooler,
    'meanmaxstartnorm': MeanStartMaxNormPooler,
    'noaffine': NoAffinePooler,
    'nemoto': NemotoPooler,
}

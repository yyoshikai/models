import torch
import torch.nn as nn
from ..models import function_config2func, register_module, get_module

@register_module
class LearnableAffine(nn.Module):
    def __init__(self, weight, bias, input_size):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size,)
        self.weight = nn.Parameter(torch.full(input_size, fill_value=float(weight)))
        self.bias = nn.Parameter(torch.full(input_size, fill_value=float(bias)))
    def forward(self, input):
        return input*self.weight+self.bias

@register_module
class BatchSecondBatchNorm(nn.Module):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=input_size, **kwargs)
    def forward(self, input):
        input = input.transpose(0, 1)
        input = self.norm(input)
        return input.transpose(0, 1)
    
# Tunnelと同じだが, 公式と同じ書き方の方が良い気がした。
@register_module
class Sequential(nn.Sequential):
    def __init__(self, args):
        layers = []
        for arg in args:
            layers.append(get_module(None, **arg))
        super().__init__(*layers)

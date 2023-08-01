import sys, os
import math
import logging
from collections import OrderedDict
from inspect import signature
from functools import partial
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def subs_vars(config, vars):
    if isinstance(config, str):
        if config in vars:
            return vars[config]
        for key, value in vars.items():
            config = config.replace(key, str(value))
        return config
    elif isinstance(config, dict):
        return Dict({label: subs_vars(child, vars) for label, child in config.items()})
    elif isinstance(config, list):
        return [subs_vars(child, vars) for child in config]
    else:
        return config

# function
def NewGELU(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) \
        * (input + 0.044715 * torch.pow(input, 3.0))))

def sigmoid(x):
    return 1/(1+math.e**(-x))

def init_config2func(layer_config):
    if type(layer_config) == str:
        name = layer_config
    elif type(layer_config) in {int, float}:
        name = layer_config
    elif layer_config == {}:
        name = 'none'
    else:
        name = layer_config.name
    if type(name) in {int, float}:
        return lambda x: nn.init.constant_(x, float(name))
    if name == 'glorot_uniform':
        return nn.init.xavier_uniform_
    elif name == 'glorot_normal':
        return nn.init.xavier_normal_
    elif name == 'he_uniform':
        return nn.init.kaiming_uniform_
    elif name == 'he_normal':
        return nn.init.kaiming_normal_
    elif name == 'uniform':
        return lambda x: nn.init.uniform_(x, layer_config.a, layer_config.b)
    elif name == 'normal':
        return lambda x: nn.init.normal_(x, layer_config.mean, layer_config.std)
    elif name in ['zero', 'zeros']:
        return nn.init.zeros_
    elif name in ['one', 'ones']:
        return nn.init.ones_
    elif name == 'none':
        return lambda x: None
    else:
        raise ValueError(f"Unsupported types of init function: {layer_config}")
function_name2func = {
    'relu': F.relu,
    'gelu': F.gelu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'newgelu': NewGELU,
    'none': lambda x: x,
    'exp': torch.exp,
    'log': torch.log,
    'log_softmax': F.log_softmax,
    'softplus': F.softplus,
}
def function_config2func(config):
    if isinstance(config, str):
        return function_name2func[config]
    elif config.type == 'affine':
        weight = config.weight if 'weight' in config else 1.0
        bias = config.bias if 'bias' in config else 0.0
        return lambda x: x*weight+bias
    else:
        return partial(function_name2func[config.pop('type')], **config)
    
# layer and tunnel
class Affine(nn.Module):
    def __init__(self, weight, bias, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.full((input_size,), fill_value=float(weight)))
        self.bias = nn.Parameter(torch.full((input_size,), fill_value=float(bias)))
    def forward(self, input):
        return input*self.weight+self.bias
class BatchSecondBatchNorm(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=input_size, **args)
    def forward(self, input):
        input = input.transpose(0, 1)
        input = self.norm(input)
        return input.transpose()
def get_layer(config, input_size):    
    if config.type == 'view':
        new_shape = []
        for size in config.shape:
            if size == 'batch_size':
                assert -1 not in new_shape, f"Invalid config.shape: {config.shape}"
                size = -1
            new_shape.append(size)
        layer = lambda x: x.view(*new_shape)
        input_size = config.shape
    elif config.type == 'slice':
        slices = (slice(*slice0) for slice0 in config.slices)
        layer = lambda x: x[slices]
        for dim, slice0 in enumerate(slices):
            if isinstance(input_size[dim], int):
                start = slice0.start if slice0.start is not None else 0
                stop = slice0.stop if slice0.start is not None else input_size[dim]
                step = slice0.step if slice0.stop is not None else 1
                input_size[dim] = (stop - start) // step
    elif config.type == 'squeeze':
        config.setdefault('dim', None)
        layer = lambda x: torch.squeeze(x, dim=config.dim)
        if config.dim == None:
            input_size = [s for s in input_size if s != 1]
        else:
            if input_size[config.dim] != 1:
                raise ValueError(f"{config.dim} th dim of size {input_size} is not squeezable.")
            size = list(input_size)[:config.dim]+list(input_size[config.dim+1:])
        input_size = size
    elif config.type in ['norm', 'layernorm', 'ln']:
        layer = nn.LayerNorm(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
    elif config.type in ['batchnorm', 'bn']:
        layer = nn.BatchNorm1d(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
    elif config.type in ['batchsecond_batchnorm', 'bsbn']:
        layer = BatchSecondBatchNorm(input_size[-2], args=config.args)
    elif config.type == "linear":
        layer = nn.Linear(input_size[-1], config.size, **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
        input_size = input_size[:-1]+[config.size]
    elif config.type == "laffine":
        layer = Affine(config.init.weight, config.init.bias, input_size[-1])
    elif config.type == "affine": # for compatibility
        layer = Affine(config.weight, config.bias, input_size[-1])
    elif config.type == "function":
        layer = function_config2func(config.function)
    elif config.type == "dropout":
        layer = nn.Dropout(**config.args)
    else:
        raise ValueError(f"Unsupported config: {config.type}")
    return layer, input_size
class Tunnel(nn.Module):
    def __init__(self, layers, input_size, logger=None): # logger for compatibility
        super().__init__()
        self.layers = []
        modules = []
        for i_layer, layer_config in enumerate(layers):
            if logger is not None:
                logger.debug(f"generating {i_layer} th layer.")
            layer, input_size = get_layer(layer_config, input_size)
            self.layers.append(layer)
            if isinstance(layer, nn.Module):
                modules.append(layer)
        self.output_size = input_size
        self.modules_ = nn.ModuleList(modules)
    def forward(self, input):
        next_input = input
        for layer in self.layers:
            next_input = layer(next_input)
        return next_input


# Model
module_type2class = {}
def get_module(logger, type, **kwargs):
    cls = module_type2class[type]
    args = set(signature(cls.__init__).parameters.keys())
    uargs = {}
    for key in list(kwargs.keys()):
        if key in args:
            uargs[key] = kwargs.pop(key)
    if len(kwargs) > 0:
        logger.warning(f"Unknown kwarg in {cls.__name__}: {kwargs}")
    return cls(**uargs)
class Model(nn.ModuleDict):
    def __init__(self, logger: logging.Logger, modules: dict, use_modules:list=None,
            omit_modules: list=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        if (use_modules is not None) and (omit_modules is not None):
            raise ValueError(f"Please specify either use_modules or omit_modules")
        mods = OrderedDict()
        modules = subs_vars(modules, {'$$model': self})
        for mod_name, mod_config in modules.items():
            if (use_modules is not None and mod_name not in use_modules) or \
                (omit_modules is not None and mod_name in omit_modules):
                continue
            logger.debug(f"Building {mod_name}...")
            mods[mod_name] = get_module(logger=logger, **mod_config)
        super().__init__(modules=mods)
        self.logger = logger
    def forward(self, batch, processes: Dict):
        for process in processes:
            type = process.pop('type') if 'type' in process else 'forward'
            if type == 'module':
                func = self[process.module]
            elif type == 'function':
                func = function_config2func(process.function)
            input_name = process.input
            if input_name is None:
                output = func(**process.kwargs)
            if isinstance(input_name, str):
                output = func(batch[input_name], **process.kwargs)
            elif isinstance(input_name, list):
                output = func(*[batch[i] for i in input_name], **process.kwargs)
            elif isinstance(input_name, dict):
                output = func(**{n: batch[i] for n, i in input_name.items()}, **process.kwargs)
            output_name = process.output
            if isinstance(output_name, str):
                batch[output_name] = output
            elif isinstance(output_name, list):
                for oname0, out in zip(output_name, output):
                    batch[oname0] = out
            else:
                raise ValueError(f'Unsupported type of output: {output_name}')
    def load(self, path, replace={}, strict=True):
        if os.path.isfile(path):
            state_dict = torch.load(path)
            for key, replace in replace.items():
                for sdict_key, sdict_value in list(state_dict.items()):
                    if sdict_key[:len(key)] == key:
                        state_dict[(replace+sdict_key[len(key):])] = sdict_value
                        del state_dict[sdict_key]
            keys = self.load_state_dict(state_dict, strict=strict)
            if len(keys.missing_keys) > 0:
                self.logger.warning("Missing keys: ")
                for key in keys.missing_keys:
                    self.logger.warning(f"  {key}")
            if len(keys.unexpected_keys) > 0:
                self.logger.warning("Unexpected keys: ")
                for key in keys.unexpected_keys:
                    self.logger.warning(f"  {key}")
        elif os.path.isdir(path):
            replace_inverse = {value: key for key, value in replace.items()}
            for mname, module in self.items():
                if mname in replace_inverse:
                    mname = replace_inverse[mname]
                mpath = f"{path}/{mname}.pth"
                if os.path.exists(mpath):
                    keys = module.load_state_dict(torch.load(mpath), strict=strict)
                    if len(keys.missing_keys) > 0:
                        self.logger.warning(f"Missing keys in {mname}: ")
                        for key in keys.missing_keys:
                            self.logger.warning(f"  {key}")
                    if len(keys.unexpected_keys) > 0:
                        self.logger.warning(f"Unexpected keys in {mname}: ")
                        for key in keys.missing_keys:
                            self.logger.warning(f"  {key}")
                else:
                    if strict:
                        raise ValueError(f"State dict file of {mname} does not exists.")
                    else:
                        self.logger.warning(f"State dict file of {mname} does not exists.")
        else:
            raise ValueError(f"Invalid file: {path}")
        

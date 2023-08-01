import os
import math
import logging
from collections import OrderedDict
from functools import partial
import numpy as np
from addict import Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import check_leftargs
from .models2 import init_config2func, function_config2func, get_layer, Tunnel
print("Use of models.py is depricated. Use 'models2.py' instead.")

# Tunnel
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

# modules
class Module(nn.Module):
    def __init__(self, logger, sizes, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)

class LayerModule(nn.Module):
    def __init__(self, logger, sizes, input, output, layer, modes=None, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
        self.input = input
        self.output = output
        self.layer, sizes[self.output] = get_layer(config=layer, input_size=sizes[self.input])
    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes: 
            return batch
        batch[self.output] = self.layer(batch[self.input])
        return batch
class TunnelModule(nn.Module):
    def __init__(self, logger, sizes, input, output, layers, modes=None, residual=False, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
        self.input = input
        self.output = output
        self.tunnel = Tunnel(logger, layers, sizes[self.input])
        self.residual = residual
        sizes[self.output] = self.tunnel.output_size
        if self.residual:
            if np.all(np.array(sizes[self.output]) != np.array(sizes[self.input])):
                raise ValueError(f"Size of output ({sizes[self.output]}) does not match size of input({sizes[self.input]}) in residual tunnel.")

    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes: 
            return batch
        batch[self.output] = self.tunnel(batch[self.input])
        return batch
component_type2class = {
    ""


}



module_type2class = {
    "tunnel": TunnelModule, 
    "layer": LayerModule
}
nn_type2class = {
    'Transformer': nn.Transformer,
    'Embedding': nn.Embedding
}
def get_module(logger, sizes, type, **kwargs):
    if type in module_type2class:
        return module_type2class[type](logger=logger, sizes=sizes, **kwargs)
    else:
        return nn_type2class[type](**kwargs)


# Model
class Model(nn.ModuleDict):
    def __init__(self, logger: logging.Logger, config: dict, sizes: dict, seed: int=0, use_modules=None,
            omit_modules=None):
        """
        Base model.

        Parameters
        ----------
        logger: logging.Logger
            logger to log messages while building model.
        config: dict
            config of names to each modules.
        seed(optional): int
        use_modules(optional): list of str
            name of modules to use. modules whose names are not in use_modules will be ignored.
        omit_modules(optional): list of str
            name of modules to ignore.
            Assignment of use_modules and omit_modules is incompatible.
        """
        if (use_modules is not None) and (omit_modules is not None):
            raise ValueError(f"Please specify either use_modules or omit_modules")
        config = config.copy()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mdict = OrderedDict()
        self.logger = logger
        for mod_name, mod_config in config.items():
            if (use_modules is not None and mod_name not in use_modules) or \
                (omit_modules is not None and mod_name in omit_modules):
                continue
            logger.debug(f"Model: generating {mod_name}")
            mdict[mod_name] = get_module(logger, sizes, **mod_config)
            logger.debug("Variables in batch:")
            logger.debug(f"  {list(sizes.keys())}")
        super().__init__(mdict)
    def forward(self, batch, mode, procedure):
        for process in procedure:
            if process.type == 'module':
                self[process.module](batch, mode)
            elif process.type == 'forward':
                inputs = process.inputs
                if isinstance(inputs, str):
                    outputs = self[process.module](batch[inputs])
                elif isinstance(inputs, list):
                    outputs = self[process.module](*[batch[i] for i in inputs])
                else:
                    outputs = self[process.module](**{key: batch[name] for key, name in inputs.items()})
                output_names = process.outputs
                if isinstance(output_names, str):
                    batch[output_names] = outputs
                else:
                    for o, oname in zip(outputs, output_names):
                        batch[oname] = o
        return batch

    def load_state_dict(self, path, no_loads_new=[], no_loads_old=[], module_names = {}, strict = True):
        """
        path: str
            path to directory or .pth path(for compatibility)
        no_loads: list

        files: {str: str}
            
        """
        if os.path.isfile(path):
            state_dict = torch.load(path)
            for key in list(state_dict.keys()):
                for mname in no_loads_old:
                    if key[:len(mname)] == mname:
                        del state_dict[key]
        for mname, module in self.items():
            self.logger.debug(f"Loading state_dict of {mname}")
            if mname in no_loads_new:
                continue
            fname = module_names[mname] if mname in module_names \
                else mname
            if os.path.isfile(path):
                sdict0 = {}
                for key, value in state_dict.items():
                    if key[:len(fname)] == fname:
                        sdict0[key[len(fname)+1:]] = value
                try:
                    module.load_state_dict(sdict0, strict=strict)
                except RuntimeError as e:
                    self.logger.error(f"---ERROR IN LOADING {mname}---")
                    self.logger.debug("Keys in original sdict: ")
                    for key, value in state_dict.items():
                        self.logger.debug(f"  {key}: {list(value.shape)}")
                    self.logger.error("Old state dict: ")
                    for key, value in sdict0.items():
                        self.logger.error(f"  {key}: {list(value.shape)}")
                    self.logger.error("New state dict: ")
                    for key, value in module.state_dict().items():
                        self.logger.error(f"  {key}: {list(value.shape)}")
                    raise e
            else:
                sdict = torch.load(f"{path}/{fname}.pth")
                try:
                    module.load_state_dict(sdict, strict=strict)
                except RuntimeError as e:
                    self.logger.error(f"---ERROR IN LOADING {mname}---")
                    self.logger.error("Original sdict:")
                    for key, value in sdict.items():
                        self.logger.error(f"  {key}: {list(value.shape)}")
                    for key, value in module.state_dict.items():
                        self.logger.error(f"  {key}: {list(value.shape)}")
                    raise e
    def save_state_dict(self, path):
        os.makedirs(path, exist_ok=True)
        for key, value in self.items():
            torch.save(value.state_dict(), f"{path}/{key}.pth")
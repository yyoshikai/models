import numpy as np
import torch

def agg(values: list, batch_dim: int):
    if isinstance(values[0], list):
        values1 = []
        for value in values:
            values1 += value
        values = values1
    elif isinstance(value[0], np.ndarray):
        values = np.concatenate(values, axis=batch_dim)
    elif isinstance(value[0], torch.Tensor):
        values = torch.cat(values, dim=batch_dim)
    else: raise ValueError
    return values

class Accumulator:
    def __init__(self, 
            usecols: list=None, 
            batch_dims: dict[str,int]={}, 
            keep_gpu: bool=False):
        self.usecols = usecols
        self.accs = None
        self.batch_dims: dict = batch_dims
        self.keep_gpu = keep_gpu

    def __call__(self, batch: dict):
        if self.accs is None:
            self.accs = { key: [] for key in 
                    (self.usecols if self.usecols is not None else batch.keys())}

        for key in self.accs:
            value = batch[key]
            if isinstance(value, torch.Tensor) and not self.keep_gpu:
                value = value.to('cpu')
            self.accs[key].append(value)

    def agg(self, index=None):
        if index in self.accs:
            index = agg(self.accs[index])



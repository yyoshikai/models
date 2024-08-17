import itertools
import numpy as np
import torch


def _max(value):
    if isinstance(value, np.ndarray):
        return np.max(value)
    elif isinstance(value, torch.Tensor):
        return torch.max(value).item()
    else:
        raise ValueError

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
            batch_dim: dict[str,int]={}, 
            keep_gpu: bool=False):
        self.usecols = usecols
        self.accs = None
        self.batch_dim: dict = batch_dim
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
        if index is not None:
            index = self.accs[index]
            if isinstance(index[0], list):
                index = list(itertools.chain.from_iterable(index))
                index = np.argsort(np.array(index))
            elif isinstance(index[0], np.ndarray):
                index = np.argsort(np.concatenate(index))
            elif isinstance(index[0], torch.Tensor):
                index = torch.argsort(torch.cat(index)).cpu().numpy()
            else: 
                raise ValueError
            assert len(index) == np.max(index)

        agg_batch = {}
        for key, acc in self.accs.items():
            batch_dim = self.batch_dim.get(key, 0)
            if isinstance(acc[0], list):
                assert batch_dim == 0
                acc = list(itertools.chain.from_iterable(acc))
                acc = [acc[i] for i in index]
            elif isinstance(acc[0], np.ndarray):
                acc = np.concatenate(acc)[(slice(None),)*batch_dim+(index,)]
            elif isinstance(acc[0], torch.Tensor):
                acc = torch.cat(acc)[(slice(None),)*batch_dim+(index,)]
            else:
                raise ValueError
            agg_batch[key] = acc
        return agg_batch





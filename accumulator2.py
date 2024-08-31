import itertools
import pickle
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

def is_concatable(acc: list, batch_dim: int):
    is_concatable = True
    shape = None
    for acc0 in acc:
        shape0 = list(acc0.shape)
        shape0.pop(batch_dim)
        if shape is None: 
            shape = shape0
        else:
            if shape != shape0:
                is_concatable = False
                break
    return is_concatable

class Accumulator:
    def __init__(self, 
            usecols: list=[], 
            batch_dim: dict[str,int]={}, 
            keep_gpu: bool=False):
        self.usecols = usecols
        self.batch_dim: dict = batch_dim
        self.keep_gpu = keep_gpu

    def init(self):
        self.accs = {col: [] for col in self.usecols}

    def __call__(self, batch: dict):

        for key in self.usecols:
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
                if is_concatable(acc):
                    acc = np.concatenate(acc)[(slice(None),)*batch_dim+(index,)]
                else:
                    transpose = list(range(acc[0].ndim))
                    transpose.pop(batch_dim)
                    transpose.insert(0, batch_dim)
                    acc_list = []
                    for acc0 in acc:
                        acc_list += list(acc0.transpose(transpose))
                    acc = acc_list
            elif isinstance(acc[0], torch.Tensor):
                if is_concatable(acc):
                    acc = torch.cat(acc)[(slice(None),)*batch_dim+(index,)]
                else:
                    transpose = list(range(acc[0].ndim))
                    transpose.pop(batch_dim)
                    transpose.insert(0, batch_dim)
                    acc_list = []
                    for acc0 in acc:
                        acc_list += list(acc0.permute(transpose))
                    acc = acc_list
            else:
                raise ValueError
            agg_batch[key] = acc
        
        return agg_batch

    def save_agg(self, save_dir, index=None):
        agg_batch = self.agg(index)

        for key, value in agg_batch.items():
            if isinstance(value, list):
                with open(f"{save_dir}/{key}.pkl", 'wb') as f:
                    pickle.dump(value, f)
            elif isinstance(value, np.ndarray):
                np.save(f"{save_dir}/{key}.npy", value)
            elif isinstance(value, torch.Tensor):
                torch.save(value, f"{save_dir}/{key}.pt")








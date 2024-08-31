import sys, os
import math
import pickle
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
logger = getLogger(__name__)

torch_name2dtype = {
    'int': torch.int,
    'long': torch.long,
    'float': torch.float,
    'bool': torch.bool,
}
name2dtype = {
    'int': int,
    'float': float,
    'bool': bool,
    'object': object,
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, name):
        self.name = name
    def collate(self, batch: dict, data: tuple[torch.Tensor], device: torch.device): # default collate
        batch[self.name] = torch.stack(data, dim=0).to(device)

class SequenceDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, name, dfs,
            path, padding_value, dtype='long', dim=1, shape=None):
        super().__init__(name)
        with open(path, 'rb') as f:
            self.sequences = pickle.load(f)
        self.lengths = np.array([len(s) for s in self.sequences], dtype=int)
        self.padding_value = padding_value
        self.dtype = torch_name2dtype[dtype]
        self.dim = dim
        if self.dim == 1:
            if shape is not None:
                self.logger.warning(f"dim is 1 and shape({shape}) is ignored.")
        else:
            self.shape = shape or []

    def __getitem__(self, index):
        return torch.tensor(self.sequences(index), dtype=self.dtype)
    
    def __len__(self):
        return len(self.sequences)

    def collate(self, batch: dict, data: tuple[torch.Tensor], device: torch.device):
        data_lens = torch.tensor([len(d) for d in data], dtype=torch.long, device=device)
        batch[self.name+'_len'] = data_lens
        if self.dim == 1:
            batch[self.name] = pad_sequence(data, batch_first=True, padding_value=self.padding_value)
        else:
            batch_sequences = torch.full((len(data), )+(torch.max(data_lens), )*self.dim+self.shape,
                fill_value=self.padding_value, dtype=self.dtype)
            for i, d in enumerate(data):
                batch_sequences[(i, )+(slice(data_lens[i]), )*self.dim] = d
            batch[self.name] = batch_sequences.to(device)

    def get_lengths(self): # For bucketing
        return self.lengths


class ArrayDataset(Dataset):
    def __init__(self, name, dfs,
                dtype='float', atype='torch', getitem_subs=False, **kwargs):
        super().__init__(name)
        data = self.init_data(dfs=dfs, **kwargs)
        if atype in ['numpy', 'np']: 
            self.data = np.array(data, dtype=name2dtype[dtype])
        elif atype == 'torch':
            self.data = torch.tensor(data, dtype=torch_name2dtype[dtype])
        else:
            raise ValueError
        self.getitem_subs = getitem_subs

    def init_data(self, dfs, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.getitem_subs:
            return self.data[index]
        else:
            return index
    def __getitems__(self, indices: list):
        return tuple(self.data[indices])

    def collate(self, batch: dict, data: tuple[np.ndarray|torch.Tensor], 
                device: torch.device):
        if isinstance(self.data[0], np.ndarray):
            data = np.stack(data, axis=0)
        else:
            data = torch.stack(data, dim=0)
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        batch[self.name] = data
        
class NdarrayDataset(ArrayDataset):
    def init_data(self, dfs, path, cols=None):
        if isinstance(path, np.ndarray):
            data = path
        else:
            data = np.load(path)
        if cols is not None: data = data[:, cols]
        return data

class SeriesDataset(ArrayDataset):
    def init_data(self, dfs, df, col):
        return dfs[df][col]

class DataFrameDataset(ArrayDataset):
    def init_data(self, dfs, df, cols=None):
        data = dfs[df]
        if cols is not None: data = data[cols]
        return data.values
    
class BitDataset(Dataset):
    def __init__(self, name, dfs, path, size, dtype='long'):
        super().__init__(name)
        self.size = size
        self.packed_array = np.load(path)
        assert len(self.packed_array[0]) == math.ceil(size/8)
        self.dtype = torch_name2dtype[dtype]
    def __getitem__(self, index):
        packed_data = self.packed_array[index]
        data = np.unpackbits(packed_data)
        return torch.tensor(data, dtype=self.dtype)

    def __len__(self):
        return len(self.packed_array)

class SparseSquareDataset(Dataset):
    def __init__(self, name, dfs, 
        padding_value: int, path_length: str, path_index: str, path_value: str, 
        dtype=None, symmetrize=False, return_sparse=False):

        super().__init__(name)
        ext = os.path.splitext(path_length)[1]
        if ext == '.npy':
            self.lengths = np.load(path_length)
        elif ext == '.pkl':
            with open(path_length, 'rb') as f:
                self.lengths = np.array(pickle.load(f))
            if self.lengths.dtype == np.object_:
                self.lengths = self.lengths.astype(int)
        else:
            raise ValueError(f"Unsupported type of path_length: {path_length}")
        
        with open(path_index, 'rb') as f:
            self.indices = pickle.load(f)
        with open(path_value, 'rb') as f:
            self.values = pickle.load(f)
        self.padding_value = padding_value
        if dtype is not None:
            self.dtype = torch_name2dtype[dtype]
        else:
            self.dtype = None
        self.symmetrize = symmetrize
        self.return_sparse = return_sparse
    
    def __getitem__(self, index: int):
        length = self.lengths[index]
        idx = torch.tensor(self.indices[index], dtype=torch.long).T # [2, n_edge]
        value = torch.tensor(self.values[index], dtype=self.dtype)
        return length, idx, value # [], [2, n_edge], [n_edge]
    
    def collate(self, batch: dict, data: tuple[tuple], device: torch.device):
        lengths, idxs, values = zip(*data)
        batch_size = len(data)

        lengths = torch.tensor(lengths, device=device)
        batch[self.name+'_len'] = lengths
        max_length = torch.max(lengths)

        ibatches = torch.cat([ torch.full((1, idx.shape[1]), fill_value=i, dtype=torch.long)
                for i, idx in enumerate(idxs)], dim=1).to(device)
        idxs_b = torch.cat(idxs, dim=1).to(device)
        idxs_b = torch.cat([ibatches, idxs_b], dim=0).to(device) # [3, n_edge]
        values_b = torch.cat(values, dim=0).to(device)
        data = torch.sparse_coo_tensor(idxs_b, values_b, 
            size=(batch_size, max_length, max_length), device=device).to_dense()
        if self.symmetrize:
            data = data + data.transpose(-1, -2)
        batch[self.name] = data

        if self.return_sparse:
            idxs_r0 = pad_sequence([idx[0] for idx in idxs],
                    batch_first=True, padding_value=0) # [B, n_edge]
            idxs_r1 = pad_sequence([idx[1] for idx in idxs],
                    batch_first=True, padding_value=1) # [B, n_edge] 240810なぜpadding_valueが違うのか？
            idxs_r = torch.stack([idxs_r0, idxs_r1], dim=-1)
            batch[self.name+'_indices'] = idxs_r
            values_r = pad_sequence(values, batch_first=True, padding_value=0)
            batch[self.name+'_values'] = values_r

    def get_lengths(self):
        return self.lengths


class GenerateDataset(Dataset):
    def __init__(self, name, dfs, feature_size, data_size, **kwargs):
        super().__init__(name)
        self.feature_size = feature_size
        self.data_size = data_size

    def __getitem__(self, index):
        return torch.randn((self.feature_size,))
    
    def __getitems__(self, indices):
        return list(torch.randn((len(indices), self.feature_size)))
    
    def __len__(self):
        return self.data_size

dataset_type2class = {
    'sequence': SequenceDataset,
    'ndarray': NdarrayDataset,
    'series': SeriesDataset,
    'dataframe': DataFrameDataset,
    'bit': BitDataset,
    'sparse_square': SparseSquareDataset
}

def register_dataset(name):
    def register_dataset_cls(cls):
        assert issubclass(cls, Dataset)
        dataset_type2class[name] = cls
        return cls
    return register_dataset_cls

def get_dataset(type, name, dfs, **kwargs) -> Dataset:
    return dataset_type2class[type](name=name, dfs=dfs, **kwargs,)

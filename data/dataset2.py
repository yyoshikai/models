import sys, os
import math
import pickle
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
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


class ArrayDataset(Dataset):
    def __init__(self, name, dfs,
                dtype, atype='torch', **kwargs):
        super().__init__(name)
        data = self.init_data(dfs=dfs, **kwargs)
        if atype in ['numpy', 'np']: 
            self.data = np.array(data, dtype=name2dtype[dtype])
        elif atype == 'torch':
            self.data = torch.tensor(data, dtype=torch_name2dtype[dtype])
        else:
            raise ValueError

    def init_data(self, dfs, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        return index
    
    def collate(self, batch: dict, data: tuple[int], device: torch.device):
        data = self.data[data]
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        batch[self.name] = data
        
class NdarrayDataset(ArrayDataset):
    def init_data(self, dfs, path, cols=None):
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
        dtype=None, symmetrize=True, return_sparse=False):
        """
        symmetrizeのデフォルトをTrueに変更(最悪間違えても大丈夫)
        """
        super().__init__(name)
        ext = os.path.splitext(path_length)[1]
        if ext == '.npy':
            lengths = np.load(path_length)
        elif ext == '.pkl':
            with open(path_length, 'rb') as f:
                lengths = np.array(pickle.load(f))
            if lengths.dtype == np.object_:
                lengths = lengths.astype(int)
        else:
            raise ValueError(f"Unsupported type of path_length: {path_length}")
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        
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
        idx = torch.tensor(self.indices[index], dtype=torch.long).T # [2, n_edge]
        value = torch.tensor(self.values[index], dtype=self.dtype)
        return index, idx, value
    def collate(self, batch: dict, data: tuple[tuple], device: torch.device):
        indices, idxs, values = zip(*data)
        ibatches = torch.cat([ torch.full((1, idx.shape[1]), fill_value=i, dtype=torch.long)
                for i, idx in enumerate(idxs)], dim=1).to(device)
        idxs = torch.cat(idxs, dim=1).to(device)
        idxs = torch.cat([ibatches, idxs], dim=0).to(device) # [3, n_edge]
        values = torch.cat(values, dim=0).to(device)
        

    def make_batch(self):
        batch_size = len(idx)
        lengths = self.lengths[idx].to(device)
        batch[self.len_name] = lengths
        max_len = torch.max(lengths)
        ibatches = []
        indices = []
        values = []
        if self.return_sparse:
            rindices = []
            rvalues = []
        for i, idx in enumerate(idx):
            index = torch.tensor(self.indices[idx], dtype=torch.long, device=device) # [n_edge, 2]
            value = torch.tensor(self.values[idx], dtype=self.dtype, device=device)
            indices.append(index)
            ibatches.append(torch.full((index.shape[0], ), fill_value=i, dtype=torch.int, device=device))
            values.append(value)
            if self.return_sparse:
                rindices.append(index) # [n_edge, 2]
                rvalues.append(value)
        ibatches = torch.cat(ibatches, dim=0)
        indices = torch.cat(indices, dim=0).T # [2, n_edges]
        indices = torch.cat([ibatches.unsqueeze(0), indices], dim=0)
        values = torch.cat(values, dim=0)
        data = torch.sparse_coo_tensor(indices, values, size=(batch_size, max_len, max_len)).to_dense()
        if self.symmetrize:
            data = data + data.transpose(-1, -2)
        batch[self.name] = data
        if self.return_sparse:
            rindices0 = pad_sequence([rindex[:,0] for rindex in rindices],
                batch_first=True, padding_value=0) # [B, n_edge]
            rindices1 = pad_sequence([rindex[:,1] for rindex in rindices], 
                batch_first=True, padding_value=1) # [B, n_edge]
            rindices = torch.stack([rindices0, rindices1], dim=-1)
            batch[self.name + '_indices'] = rindices
            rvalues = pad_sequence(rvalues, batch_first=True, padding_value=0) # [B, n_edge]
            batch[self.name + '_values'] = rvalues
    
    
    
        pass
    
    
    def collate(self, batch: dict, data: tuple[int], device: torch.device):
        pass






    

dataset_type2class = {
    'sequence': SequenceDataset,
    'ndarray': NdarrayDataset,
    'series': SeriesDataset,
    'dataframe': DataFrameDataset,
    'bit': BitDataset,
    'sparse_square': SparseSquareDataset
}

def get_dataset(type, **kwargs) -> Dataset:
    return dataset_type2class[type](**kwargs)

class Datasets(Dataset):
    def __init__(self, datasets: dict[str, dict], dfs: dict[str, dict] = {}):
        dfs = {name: pd.read_csv(**df) for name, df in dfs}
        self.datasets = [
            get_dataset(name=name, dfs=dfs, **dataset) for name, dataset in datasets.items()
        ]
    
    def __getitem__(self, index):
        return (dataset[index] for dataset in self.datasets)

    def collate(self, datas: list):
        batch = {}
        for dataset, data in zip(self.datasets, zip(*datas)):
            dataset.collate(batch, data)
        return batch

class NormalSampler(Sampler):
    def __init__(self, datasets: Datasets, batch_size: int, 
                ):
        pass

class DataLoader0(torch.utils.data.DataLoader):
    def __init__(self, datasets, dfs, batch_sampler, num_workers=0, ):
        datasets = Datasets(datasets, dfs)
        batch_sampler = get_batch_sampler(datasets=datasets, **batch_sampler)

        super().__init__()
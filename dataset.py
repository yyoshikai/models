import sys, os
import math
import pickle
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, BatchSampler
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
        if getitem_subs: # こういう実装してみたが, 分かりやすいか？
            self.getitemer = lambda index: self.data[index]
            if isinstance(self.data, np.ndarray):
                self.collator = lambda data: np.stack(data, axis=0)
            else:
                self.collator = lambda data: torch.stack(data, dim=0)
        else:
            self.getitemer = lambda index: index
            self.collator = lambda data: self.data[data]

    def init_data(self, dfs, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.getitemer(index)
    
    def collate(self, batch: dict, data: tuple[int], device: torch.device):
        data = self.collator(data)
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

class Datasets(Dataset):
    def __init__(self, datasets: dict[str, dict], dfs: dict[str, dict] = {}):
        dfs = {name: pd.read_csv(**df) for name, df in dfs}
        self.datasets = \
            {name: get_dataset(name=name, dfs=dfs, **dataset) for name, dataset in datasets.items()}
    
    def __getitem__(self, index):
        return (dataset[index] for dataset in self.datasets.values())

    def collate(self, datas: list):
        batch = {}
        for dataset, data in zip(self.datasets.values(), zip(*datas)):
            dataset.collate(batch, data)
        return batch
    
    def __len__(self):
        for dataset in self.datasets.values():
            try:
                return len(dataset)
            except NotImplementedError:
                pass
        return NotImplementedError

class RandomSampler(torch.utils.data.RandomSampler):
    def __init__(self, datasets, 
            seed: int, **kwargs):
            generator = torch.Generator()
            generator.manual_seed(seed)
            super().__init__(datasets, generator=generator, **kwargs)

class NormalSampler(BatchSampler):
    def __init__(self, datasets, 
            seed: int,
            batch_size: int,
            drop_last: bool = False,
            replacement: bool=False,
            num_samples=None):
        
        sampler = RandomSampler(datasets, seed, replacement=replacement, 
                num_samples=num_samples)
        super().__init__(sampler, batch_size, drop_last)

class BucketSampler(Sampler):
    def __init__(self, datasets: Datasets, 
            seed: int, 
            buckets: dict[str, list[float]],
            batch_sizes: list):
        self.batch_sizes = batch_sizes
        self.rstate = np.random.default_rng(seed)
        d2b = None
        for dset_name, bins in buckets.items():
            lengths = datasets.datasets[dset_name].get_lengths()
            d2b0 = np.digitize(lengths, bins) - 1
            if d2b is None: 
                d2b = d2b0
            else:
                d2b = np.maximum(d2b, d2b0)
        self.d2b = d2b

    def __iter__(self):
        idxs = []
        for ib, batch_size in enumerate(self.batch_sizes):
            bucket_idxs = np.where(self.d2b == ib)[0]
            if len(bucket_idxs) == 0: continue
            self.rstate.shuffle(bucket_idxs)
            idxs += [bucket_idxs[i:i+batch_size] for i in range(0, len(bucket_idxs), batch_size)]
        self.rstate.shuffle(idxs)
        return iter(idxs)
    
    def __len__(self):
        l = 0
        for ib, batch_size in enumerate(self.batch_sizes):
            bucket_idxs = np.where(self.d2b == ib)[0]
            l += math.ceil(len(bucket_idxs)/batch_size)
        return l

class ChunkSampler(Sampler):
    def __init__(self, datasets: Datasets,
            seed: int, length_data: str, batch_size: int, last: str, shuffle_chunk: bool):
        self.length_data = datasets.datasets[length_data]
        self.batch_size = batch_size
        assert last in [None, 'drop', 'refill']
        self.last = last
        self.shuffle_chunk = shuffle_chunk
        self.rstate = np.random.default_rng(seed)


    def __iter__(self):
        chunk_idxs = np.arange(len(self.length_data))
        if self.shuffle_chunk:
            self.rstate.shuffle(chunk_idxs)
        for cidx in chunk_idxs:
            data_idxs = np.arange(self.length_data[cidx])
            self.rstate.shuffle(data_idxs)
            if self.last == 'drop':
                data_idxs = data_idxs[:-len(data_idxs)%self.batch_size]
            elif self.last == 'refill':
                data_idxs = np.concatenate(data_idxs, data_idxs[])

batch_sampler_type2class = {
    'normal': NormalSampler, 
    'bucket': BucketSampler,
}
def get_batch_sampler(type, datasets, **kwargs):
    return batch_sampler_type2class[type](datasets=datasets, **kwargs)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, datasets, dfs, batch_sampler, **kwargs):
        datasets = Datasets(datasets, dfs)
        batch_sampler = get_batch_sampler(datasets=datasets, **batch_sampler)

        super().__init__(dataset=datasets, batch_sampler=batch_sampler, 
                collate_fn=datasets.collate, **kwargs)


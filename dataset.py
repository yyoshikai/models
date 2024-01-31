"""
240112 DataLoaderのdatasetsのconfigをupdateするように変更
240126 NormalDatasetのget_idxsを修正(最初が空の配列になっていた。numpy.splitの仕様)
"""

import sys, os
import inspect
import itertools
from copy import deepcopy

import yaml
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from addict import Dict
from .utils import check_leftargs

class DataLoader:
    def __init__(self, logger, datasets, seed, device, checkpoint=None, **kwargs):
        """
        logger: logger
        datasets: List
        - dfs:
            (df_name): dict
              Input for pd.read_csv
          datasets:
            (data_name): dict
              Input for get_dataset
        seed: int
        checkpoint: str or None
        """
        check_leftargs(self, logger, kwargs)
        if not isinstance(datasets, list):
            datasets = [datasets]

        dset_configs = []
        config = Dict()
        for dset_config in datasets:
            config.update(dset_config)
            cur_config = deepcopy(config)
            dset_configs.append(cur_config)
        
        self.dset_configss = dset_configs
        for dset_config in self.dset_configss:
            for df_config in dset_config.dfs.values():
                assert ('path' in df_config) != ('filepath_or_buffer' in df_config) # 240112 datasetsのupdate機能のため念のため
                if 'path' in df_config:
                    df_config['filepath_or_buffer'] = df_config.pop('path')
        self.n_dset = len(datasets)
        self.i_current_idx = 0
        self.i_dset = 0
        self.epoch = self.step = 0
        self.current_idxs = None
        self.rstate = np.random.RandomState(seed=seed)
        self.logger = logger
        self.device = device
        self.cur_dsets = None
        # load chekcpoint
        if checkpoint is not None:
            with open(f"{checkpoint}/config.yaml") as f:
                config = Dict(yaml.load(f, yaml.Loader))
            self.i_dset = config.i_dset
            self.i_current_idx = config.i_current_idx
            self.epoch = config.epoch
            self.step = config.step
            with open(f"{checkpoint}/current_idxs.pkl", 'rb') as f:
                self.current_idxs = pickle.load(f)
            with open(f"{checkpoint}/rstate.pkl", 'rb') as f:
                self.rstate.set_state(pickle.load(f))
        self.load_datasets()
    def load_datasets(self):
        del self.cur_dsets
        gc.collect()
        dfs = {}
        for df_name, df_config in self.dset_configss[self.i_dset].dfs.items():
            self.logger.info(f"Loading {df_config.filepath_or_buffer} ...")
            dfs[df_name] = pd.read_csv(**df_config)
        self.cur_dsets = {name: get_dataset(logger=self.logger, name=name, dfs=dfs, **dset_config) 
            for name, dset_config in self.dset_configss[self.i_dset].datasets.items()}
        del dfs
        self.i_cur_dsets = self.i_dset
        
    def get_batch(self, batch=None):
        if self.i_cur_dsets != self.i_dset:
            self.load_datasets()
        if self.current_idxs is None:
            self.current_idxs = self.get_idxs(self.cur_dsets)
        idx = self.current_idxs[self.i_current_idx].astype(int)
        if batch is None: batch = {}
        batch['idx'] = idx
        for dset in self.cur_dsets.values():
            dset.make_batch(batch, idx, self.device)
        batch['batch_size'] = len(batch['idx'])
        self.i_current_idx += 1
        self.step += 1
        if self.i_current_idx == len(self.current_idxs):
            self.i_current_idx = 0
            self.current_idxs = None
            self.i_dset = (self.i_dset+1)%self.n_dset
            if self.i_dset == 0:
                self.epoch += 1
        return batch
    def __iter__(self):
        self.epoch = self.i_dset = self.i_current_idx = 0
        # self.current_idxs = None
        # shuffleをするかどうか?
        while self.epoch == 0:
            yield self.get_batch()
    def get_idxs(self, dsets):
        raise NotImplementedError    
    def checkpoint(self, path_checkpoint):
        os.makedirs(path_checkpoint)
        config = {
            'i_dset': self.i_dset, 
            'i_current_idx': self.i_current_idx,
            'epoch': self.epoch,
            'step': self.step, 
        }
        with open(f"{path_checkpoint}/config.yaml", 'w') as f:
            yaml.dump(config, f)
        with open(f"{path_checkpoint}/rstate.pkl", 'wb') as f:
            pickle.dump(self.rstate.get_state(), f)
        with open(f"{path_checkpoint}/current_idxs.pkl", 'wb') as f:
            pickle.dump(self.current_idxs, f)

class NormalDataLoader(DataLoader):
    def __init__(self, logger, device, datasets, seed, batch_size, checkpoint=None, **kwargs):
        super().__init__(logger=logger, datasets=datasets, seed=seed,
            device=device, checkpoint=checkpoint, **kwargs)
        self.batch_size = batch_size
        if not isinstance(datasets, list): datasets = [datasets]
        self.dset_name0 = list(datasets[0].datasets.keys())[0]
    def get_idxs(self, dsets):
        dset_size = len(dsets[self.dset_name0])
        idxs = np.arange(dset_size, dtype=int)
        self.rstate.shuffle(idxs)
        idxs = np.split(idxs, range(self.batch_size, dset_size, self.batch_size))
        with open("idxs.pkl", 'wb') as f:
            pickle.dump(idxs, f)
        return idxs

class BucketDataLoader(DataLoader):
    def __init__(self, logger, device, datasets, seed, bucket_dset, checkpoint=None, 
        bin_linspace=None, bins=None, add_lower_margin=True, add_upper_margin=True,
        batch_size=None, num_tokens=None, num_tokens_dim=None, max_batch_size=None, **kwargs):
        """
        num_tokensを指定すると, max_lenに応じてbatch_sizeを変えるようにする。
        
        Parameters
        ----------
        bucket_dset: str
            name of dataset which bucketing is based on.
        bucket_linspace: Optional[tuple(int, int, int)]
            np.linspace(*bin_linspace) is used as bins
        bins: List[int]
            bins of bucket.
            bucket[i]: bins[i] <= length < bins[i+1]
        add_lower_margin: bool
        add_upper_margin: bool
        batch_size: Optional[int or List[int]]

        num_tokens: Optional, int
        num_tokens_dim: Optional, int
            batch_size*(length**num_tokens_dim) is restricted to num_tokens
        """
        super().__init__(logger=logger, datasets=datasets, seed=seed,
            device=device, checkpoint=checkpoint, **kwargs)
        # check args
        if (bin_linspace is None) == (bins is None):
            raise ValueError(f"Either bin_linspace({bin_linspace}) XOR bins({bins}) must be specified")
        if (batch_size is None) == (num_tokens is None):
            raise ValueError(f"Either batch_size({batch_size}) XOR num_tokens({num_tokens}) must be specified.")
        if batch_size is not None:
            assert num_tokens_dim is None, "When batch size is specified, num_tokens_dim must not be specified."
            assert max_batch_size is None, "When batch size is specified, max_batch_size must not be specified."
        else:
            if num_tokens_dim is None: num_tokens_dim = 1
            if max_batch_size is None: max_batch_size = float('inf')

        self.buckets = [None]*len(self.dset_configss)
        self.bucket_dset = bucket_dset

        # calc bucket bins
        if bin_linspace is not None:
            bins = list(np.linspace(*bin_linspace))
        if add_lower_margin and (len(bins) == 0 or bins[0] > 0):
            bins.insert(0, 0)
        if add_upper_margin and (len(bins) == 0 or bins[-1] < float('inf')):
            bins.append(float('inf'))
        self.bins = bins
        self.n_bucket = len(self.bins) - 1

        # calc batch sizes
        self.num_tokens = num_tokens
        self.num_tokens_dim = num_tokens_dim
        if batch_size is not None:
            if isinstance(batch_size, list):
                assert len(batch_size) == self.n_bucket
                self.batch_sizes = batch_size
            else:
                self.batch_sizes = [batch_size]*(len(self.bins)-1)
        else:
            self.batch_sizes = [min(int(num_tokens//(np.ceil(sup_len)-1)**num_tokens_dim), max_batch_size)
                 for sup_len in self.bins[1:]]
    def get_idxs(self, dsets):
        lengths = dsets[self.bucket_dset].lengths
        max_len = torch.max(lengths)
        amax = torch.argmax(lengths)
        ibs = np.digitize(lengths, self.bins) - 1
        batch_sizes = self.batch_sizes
        if self.num_tokens is not None and self.bins[-1] == float('inf'):
            batch_sizes[-1] = int(self.num_tokens//torch.max(lengths).item()**self.num_tokens_dim)
        idxs = []
        for ib, batch_size in enumerate(self.batch_sizes):
            bucket_idxs = np.where(ibs == ib)[0]
            if len(bucket_idxs) == 0: continue
            self.rstate.shuffle(bucket_idxs)
            idxs += [bucket_idxs[i:i+batch_size] for i in range(0, len(bucket_idxs), batch_size)]
        idxs = np.array(idxs, dtype=object)
        self.rstate.shuffle(idxs)
        return idxs

dataloader_type2class = {
    'normal': NormalDataLoader,
    'bucket': BucketDataLoader
}

def get_dataloader(type, **kwargs) -> DataLoader:
    return dataloader_type2class[type](**kwargs)

class Dataset:
    def __init__(self, logger, name, dfs, **kwargs):
        check_leftargs(self, logger, kwargs)
        self.name = name
        pass
    def make_batch(self, batch, idx, device):
        """
        Parameters
        ----------
        batch: dict
            dict into which batch element is to be input.
            ['idxs']: indices in dataset
        
        """
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

    def calc_split(self, dfs, df, col, idx=None):
        split: np.ndarray = dfs[df][col].values
        if idx is not None:
            if isinstance(idx, list):
                idx = set(idx)
                split = np.frompyfunc(lambda x: x in idx, nin=1, nout=1)(split).astype(bool)
            else:
                split = split == idx
        else:
            split = split.astype(bool)
        return split


torch_name2dtype = {
    'int': torch.int,
    'long': torch.long,
    'float': torch.float,
    'bool': torch.bool,
}
numpy_name2dtype = {
    'int': int,
    'float': float,
    'bool': bool,
}
class StringDataset(Dataset):
    def __init__(self, logger, name, dfs, 
            padding_value, list=None, path_list=None,
            len_name=None, shape=[], dtype='long', dim=1, split=None, **kwargs):
        """
        Parameters
        ----------
        padding_value(int): Pad token.
        list(list): List of raw dataset.
        path_list(str): Path to pickle file of list.
            Either 'list' or 'path_list' must be specified.
        len_name(Optional, str): Name of string length in batch
        shape(list of int): Additional shape of each datapoint.
        dtype(str): Name of dtype. Must be in torch_name2dtype
        dim: Dimension of variable length. 
        split: dict, optional
            Input for self.calc_split()
        
        Shape of each data in list should be [length, ...(dim), length, *shape] 
        """
        super().__init__(logger, name, dfs, **kwargs)
        if (list is None) == (path_list is None):
            raise ValueError(f"Either list({list}) XOR path_list({path_list}) has to be specified.")
        self.len_name = len_name or f"{self.name}_len"
        # Load str_list
        if list is not None:
            self.str_list = list
        else:
            logger.info(f"Loading {path_list} ...")
            with open(path_list, 'rb') as f:
                self.str_list = pickle.load(f)
        if split:
            split = self.calc_split(dfs, **split)
            self.str_list = list(itertools.compress(self.str_list, split))

        self.lengths = torch.tensor([len(string) for string in self.str_list], 
            dtype=torch.long)
        self.shape = tuple(shape)
        self.dtype = torch_name2dtype[dtype]
        self.dim = dim
        logger.info(f"Max length of {name}: {torch.max(self.lengths)}")

        # Other settings
        self.padding_value = padding_value

    def make_batch(self, batch, idx, device):
        n = len(idx)
        batch_lengths = self.lengths[idx].to(device)
        batch[self.len_name] = batch_lengths
        batch_strings = torch.full((n, )+(torch.max(batch_lengths), )*self.dim+self.shape,
            fill_value=self.padding_value, dtype=self.dtype)
        for i, idx in enumerate(idx):
            batch_strings[(i, )+(slice(batch_lengths[i]), )*self.dim] = torch.tensor(self.str_list[idx], dtype=self.dtype)
        batch[self.name] = batch_strings.to(device)
    def __len__(self):
        return len(self.str_list)

class ArrayDataset(Dataset):
    def __init__(self, logger, name, dfs, dtype, atype='torch', **kwargs):
        super().__init__(logger, name, dfs, **kwargs)
        self.type = atype
        if self.type in ['numpy', 'np']:
            self.type = 'numpy'
        # check dtype  
        if self.type == 'torch':
            self.dtype = torch_name2dtype[dtype]
        elif self.type == 'numpy':
            self.dtype = numpy_name2dtype[dtype]
        else:
            raise ValueError(f"Unsupported atype: {atype}")
        self.array = None
    def make_batch(self, batch, idx, device=None):
        item = self.array[idx]
        if device is not None:
            item = item.to(device)
        batch[self.name] = item
    def __len__(self):
        return len(self.array)

class NdarrayDataset(ArrayDataset):
    def __init__(self, logger, name, dfs, dtype, path, cols=None, atype='torch',split=None,  **kwargs):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        ext_path = os.path.splitext(path)[-1][1:]
        if ext_path in ['npy', 'npz']:
            array = np.load(path)
            if self.type == 'torch':
                array = torch.tensor(array)
        elif ext_path in ['pt']:
            array = torch.load(path)
            if self.type == 'numpy':
                array = array.numpy()
        else:
            raise ValueError(f"Unsupported type of ndarray: {path}")
        if self.type == 'torch':
            array = array.to(self.dtype)
        else:
            array = array.astype(self.dtype)
        if cols is not None:
            array = array[:, cols]
        if split:
            split = self.calc_split(dfs, **split)
            array = array(split)
        self.array = array
        self.size = ['batch_size'] + list(self.array.shape[1:])

class SeriesDataset(ArrayDataset):
    def __init__(self, logger, name, dfs, 
        df, dtype, col, atype='torch', split=None, **kwargs):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        array = dfs[df][col].values
        if split:
            split = self.calc_split(dfs, **split)
            array = array[split]
        if self.type == 'torch':
            self.array = torch.tensor(array, dtype=self.dtype)
        elif self.type == 'numpy':
            self.array = array.astype(self.dtype)
class DataFrameDataset(ArrayDataset):
    def __init__(self, logger, name, dfs,
        df, dtype, cols=None, atype='torch', split=None, **kwargs):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        if cols is None: cols = dfs[df].columns
        array = dfs[df][cols].values
        if split:
            split = self.calc_split(dfs, **split)
            array = array[split]
        if self.type == 'torch':
            self.array = torch.tensor(array, dtype=self.dtype)
        elif self.type == 'numpy':
            self.array = array.astype(self.dtype)

class SparseSquareDataset(Dataset):
    def __init__(self, logger, name, dfs, 
        padding_value, path_length,
        path_index, path_value, len_name=None, dtype=None, split=None, symmetrize=False, **kwargs):
        super().__init__(logger, name, dfs, **kwargs)
        """
        Parameters to write
        -------------------
        padding_value(int): Pad token.
        path_length(str): Path to pickle or npy file or of list of length of each square.
            File data shuld be list(int) or np.ndarray(int)[]
        path_index(str): path to pickle file of list of 
            File data should be list(np.ndarray[n_entry, 2])
        path_value(str): Path to pickle file of list
            File data should be list(np.ndarray[n_entry])
        """
        self.len_name = len_name or f"{self.name}_len"
        if split:
            raise NotImplementedError("Splitting for SparseSquareDataset is not defined.")
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
    def make_batch(self, batch, idx, device):
        batch_size = len(idx)
        lengths = self.lengths[idx].to(device)
        batch[self.len_name] = lengths
        max_len = torch.max(lengths)
        ibatches = []
        indices = []
        values = []
        for i, idx in enumerate(idx):
            index = torch.tensor(self.indices[idx], dtype=torch.int, device=device) # [n_edge, 2]
            indices.append(index)
            ibatches.append(torch.full((index.shape[0], ), fill_value=i, dtype=torch.int, device=device))
            values.append(torch.tensor(self.values[idx], dtype=self.dtype, device=device))
        ibatches = torch.cat(ibatches, dim=0)
        indices = torch.cat(indices, dim=0).T # [2, n_edges]
        indices = torch.cat([ibatches.unsqueeze(0), indices], dim=0)
        values = torch.cat(values, dim=0)
        data = torch.sparse_coo_tensor(indices, values, size=(batch_size, max_len, max_len)).to_dense()
        if self.symmetrize:
            data = data + data.transpose(-1, -2)
        batch[self.name] = data

    def __len__(self):
        return len(self.lengths)
    
class GenerateDataset(Dataset):
    def __init__(self, logger, name, dfs, feature_size, data_size, **kwargs):
        super().__init__(logger, name, dfs,**kwargs)
        self.feature_size = feature_size
        self.data_size = data_size
    def make_batch(self, batch, idx, device):
        batch[self.name] = torch.randn(len(idx), self.feature_size, device=device)
    def __len__(self):
        return self.data_size

dataset_type2class = {
    'string': StringDataset, 
    'ndarray': NdarrayDataset,
    'series': SeriesDataset,
    'dataframe': DataFrameDataset,
    'sparse_square': SparseSquareDataset,
    'generate': GenerateDataset
}

# Load from other modules
# from .datasets.moleculedataset import MoleculeGraphDataset
# dataset_type2class['molecule_graph'] = MoleculeGraphDataset

def get_dataset(type, **kwargs):
    return dataset_type2class[type](**kwargs)
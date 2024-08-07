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

class Dataset(torch.utils.data.Dataset):
    def collate(self, batch: dict, data: tuple, name: str):
        raise NotImplementedError

class SequenceDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, name, dfs,
            path, padding_value, dtype='long', dim=1, shape=None):
        self.name = name
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

    def collate(self, batch: dict, data: tuple[torch.Tensor]):
        data_lens = torch.tensor([len(d) for d in data], dtype=torch.long)
        batch[self.name+'_len'] = data_lens
        if self.dim == 1:
            batch[self.name] = pad_sequence(data, batch_first=True, padding_value=self.padding_value)
        else:
            batch_sequences = torch.full((len(data), )+(torch.max(data_lens), )*self.dim+self.shape,
                fill_value=self.padding_value, dtype=self.dtype)
            for i, d in enumerate(data):
                batch_sequences[(i, )+(slice(data_lens[i]), )*self.dim] = d
            batch[self.name] = batch_sequences


class ArrayDataset(Dataset):
    def __init__(self, name):
        self.name = name
        self.data = None

    def __getitem__(self, index):
        return index
    
    def collate(self, batch: dict, data: tuple[int]):
        return self.data[data]


dataset_type2class = {
    'sequence': SequenceDataset,
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
    def __init__(self, dataset: Dataset):
        pass



import torch

class DataLoader0(torch.utils.data.DataLoader):
    def __init__(self, datasets, dfs, batch_sampler, num_workers=0, ):
        datasets = Datasets(datasets, dfs)
        batch_sampler = get_batch_sampler(batch_sampler)

        super().__init__()
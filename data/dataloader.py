import numpy as np
import pandas as pd
import torch

from models.data.dataset import Dataset, NdarrayDataset, get_dataset 
from models.data.sampler import get_batch_sampler

class Datasets(Dataset):
    def __init__(self, datasets: dict[str, dict], dfs: dict[str, dict] = {}, index=True):
        dfs = {name: pd.read_csv(**df) for name, df in dfs}
        self.datasets = \
            {name: get_dataset(name=name, dfs=dfs, **dataset) for name, dataset in datasets.items()}
        if index:
            self.datasets['index'] = NdarrayDataset('index', dfs, dtype='int', atype='numpy', 
                    path=np.arange(len(self), int))

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


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, datasets, dfs, batch_sampler, **kwargs):
        datasets = Datasets(datasets, dfs)
        batch_sampler = get_batch_sampler(datasets=datasets, **batch_sampler)

        super().__init__(dataset=datasets, batch_sampler=batch_sampler, 
                collate_fn=datasets.collate, **kwargs)


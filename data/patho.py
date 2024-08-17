from logging import getLogger
import bisect
logger = getLogger(__name__)

import numpy as np
import torch
try:
    from openslide import OpenSlide
except ImportError:
    logger.info("OpenSlide is not installed.")
    OpenSlide = None

from ..data import Dataset, register_dataset, get_dataset

@register_dataset('file_list')
class FileListDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, name, dfs, paths, pindex_paths, 
                patch_size=256):
        super().__init__(name)
        if OpenSlide is None:
            logger.error("OpenSlide is not installed!")
            raise ImportError
        self.paths = get_dataset(name='paths', dfs=dfs, **paths, 
                dtype='object', atype='numpy', getitem_subs=True) # ArrayDatasetに限定してしまうのでよくない？
        self.pindex_paths = get_dataset(name='pindex_paths', dfs=dfs, **pindex_paths,
                dtype='object', atype='numpy', getitem_subs=True)
        self.logger.info("Loading pindices...")
        self.pindices = []
        lengths = []
        for i in range(len(self.paths)):
            pindex_path = self.pindex_paths[i]
            pindex = np.load(pindex_path)
            self.pindices.append(pindex)
            lengths.append(len(pindex))
        self.logger.info("Loaded.")
        self.lengths = np.array(lengths, dtype=int)
        self.length_sum = np.cumsum(self.lengths)
        self.patch_size = patch_size

    def __len__(self):
        return self.length_sum[-1]

    def __getitem__(self, index):
        file_idx = bisect.bisect_right(self.length_sum, index)
        patch_idx = index - (self.length_sum[file_idx-1] if file_idx >= 1 else 0)
        
        wsi = OpenSlide(self.paths[file_idx])
        py, px = self.pindices[file_idx][patch_idx]
        patch = wsi.read_region(location=(px*self.patch_size, py*self.patch_size),
                level=1, size=(self.patch_size, self.patch_size))
        return patch








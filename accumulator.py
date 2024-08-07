import sys, os
import pickle
import numpy as np
import torch

from .utils import EMPTY

class NumpyAccumulator:
    def __init__(self, input, batch_dim=0):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        batch_dim: [int; default=0] Dimension of batch
        """
        self.input = input
        self.converter = None
        self.batch_dim = batch_dim
    def init(self):
        self.accums = []
    def accumulate(self, indices=None):
        accums = np.concatenate(self.accums, axis=self.batch_dim)
        if indices is not None:
            accums = accums[indices]
        return accums
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + ".npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.accumulate(indices=indices))
    def __call__(self, batch):
        input = batch[self.input]
        if self.converter is None:
            if isinstance(input, np.ndarray):
                self.converter = EMPTY
            elif isinstance(input, torch.Tensor):
                self.converter = lambda x: x.cpu().numpy()
            else:
                raise ValueError(f"{self.__class__}: Unsupported input: {type(input)}")
        self.accums.append(self.converter(batch[self.input]))
class ListAccumulator:
    def __init__(self, input, batch_dim=None):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        """
        self.input = input
        self.batch_dim = batch_dim
        self.converter = None
    def init(self):
        self.accums = []
    def accumulate(self, indices=None):
        if indices is not None:
            accums = np.array(self.accums, dtype=object)
            return accums[indices].tolist()
        else:
            return self.accums
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + '.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path_without_ext}.pkl", 'wb') as f:
            pickle.dump(self.accumulate(indices=indices), f)
    def __call__(self, batch):

        input = batch[self.input]
        if self.converter is None:
            if isinstance(input, list):
                assert self.batch_dim is None, f"batch_dim cannot be defined when data is list"
                self.converter = EMPTY
            else:
                if self.batch_dim is None: self.batch_dim = 0
                if isinstance(input, torch.Tensor):
                    if self.batch_dim == 0:
                        self.converter = lambda x: list(x.cpu().numpy())
                    else:
                        self.converter = lambda x: list(x.transpose(self.batch_dim, 0).cpu().numpy())
                elif isinstance(input, np.ndarray): 
                    if self.batch_dim == 0:
                        self.converter = lambda x: list(x)
                    else:
                        self.converter = lambda x: list(x.swapaxes(0, self.batch_dim))
                else:
                    raise ValueError(f"{self.__class__}: Unsupported input: {type(input)}")
        self.accums += self.converter(batch[self.input])

accumulator_type2class = {
    'numpy': NumpyAccumulator,
    'list': ListAccumulator
}
def get_accumulator(type, **kwargs):
    return accumulator_type2class[type](**kwargs)
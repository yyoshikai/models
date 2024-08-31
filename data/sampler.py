import torch
from torch.utils.data import Sampler, BatchSampler
from models.data.dataloader import Datasets


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

            last_size = len(data_idxs)%self.batch_size
            if self.last == 'drop':
                data_idxs = data_idxs[:-last_size]
            elif self.last == 'refill':
                data_idxs = np.concatenate([data_idxs, 
                    data_idxs[:self.batch_size-last_size]])
            
            for s in range(0, len(data_idxs), self.batch_size):
                yield data_idxs[s:s+self.batch_size]
                

batch_sampler_type2class = {
    'normal': NormalSampler, 
    'bucket': BucketSampler,
}
def get_batch_sampler(type, datasets, **kwargs):
    return batch_sampler_type2class[type](datasets=datasets, **kwargs)
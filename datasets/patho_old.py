import concurrent.futures as cf

import numpy as np
import pandas as pd
import torch

from ..dataset import Dataset, register_dataset
try:
    from openslide import OpenSlide
except Exception:
    pass


@register_dataset('tggate')
class TGGATEDataset(Dataset):
    def __init__(self, logger, name, dfs, 
            pdir, n_sample, seed, split = None, no_sample_patch = False):
        """
        各WSIから1つずつサンプリングするデータセット。
        一般的なものにする予定。
        - int_to_floatはデフォルトでtrueになっている。
        
        Parameters
        ----------
        pdir: preprocessのパス。
        ppi: 1WSIあたりのパッチ数。
        split: Input to self.calc_split
        """
        super().__init__(logger, name, dfs)

        self.pdir = pdir
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)
        self.cids = dfcase.index.values
        if split:
            split = self.calc_split(dfs, **split)
        self.cids = self.cids[split]
        self.n_sample = n_sample
        self.no_sample_patch = no_sample_patch
        self.rng = np.random.default_rng(seed)

    def make_batch(self, batch, idx, device):
        bytesize = np.dtype('uint8').itemsize
        patches = []
        for i in idx:
            cid = self.cids[i]
            patch_idx = 0 if self.no_sample_patch else self.rng.choice(self.n_sample)
            patch = np.fromfile(f"{self.pdir}/sample_patch_agg/{cid}.npy", 
                count=256*256*3, offset=patch_idx*256*256*3*bytesize, 
                dtype=np.uint8).reshape(256, 256, 3)
            patches.append(patch)
        patches = np.stack(patches, axis=0) # [B, 256, 256, 3]
        patches = torch.tensor(patches.transpose(0, 3, 1, 2), device=device)
        patches = patches.to(torch.float) / 256
        batch[self.name] = patches
    
    def __len__(self):
        return len(self.cids)



@register_dataset('tggate_tc')
class TGGATETCDataset(Dataset):
    def __init__(self, logger, name, dfs, 
            pdir, diffwsi, n_sample, seed, int_to_float):
        super().__init__(logger, name, dfs)
        self.diffwsi = diffwsi
        self.rng = np.random.default_rng(seed)
        self.pdir = pdir
        self.n_sample = n_sample
        self.int_to_float = int_to_float

        # data
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)

        ## highに限定
        cond2high = {}
        conds = []
        doses = []
        for cond in dfcase['COND']:
            cond = cond.split(';')
            dose = int(cond[4])
            cond = ';'.join(cond[:4])
            if cond not in cond2high:
                cond2high[cond] = dose
            else:
                cond2high[cond] = max(cond2high[cond], dose)
            conds.append(cond)
            doses.append(dose)
        dfcase['COND'] = conds
        dfcase['DOSE'] = doses
        is_highest = []
        for cond, dose in zip(conds, doses):
            is_highest.append((cond2high[cond] == dose) and (dose != 0))
        self.dfcase_high = dfcase[is_highest]

        self.cond2cont_cids = {}
        self.cond2high_cids = {}
        for cond in np.unique(conds):
            dfcase_cond_cont = dfcase[(dfcase['COND'] == cond) & (dfcase['DOSE'] == 0)]
            self.cond2cont_cids[cond] = dfcase_cond_cont.index.values
            dfcase_high_cont = self.dfcase_high[self.dfcase_high['COND'] == cond]
            self.cond2high_cids[cond] = dfcase_high_cont.index.values

    def make_batch(self, batch, idx, device):

        patches0 = []
        patches1 = []
        cpatches = []
        for i in idx:
            cid = self.dfcase_high.index[i]
            cond = self.dfcase_high['COND'][cid]
            patches0.append(np.load(f"{self.pdir}/sample_patch/patch/{cid}/{self.rng.choice(self.n_sample)}.npy"))
            if self.diffwsi:
                cid1 = self.rng.choice(self.cond2high_cids[cond])
            else:
                cid1 = cid
            patches1.append(np.load(f"{self.pdir}/sample_patch/patch/{cid1}/{self.rng.choice(self.n_sample)}.npy"))

            ccid0 = self.rng.choice(self.cond2cont_cids[cond])
            cpatches.append(np.load(f"{self.pdir}/sample_patch/patch/{ccid0}/{self.rng.choice(self.n_sample)}.npy"))
        patches = np.stack(patches0+patches1+cpatches, axis=0) # [B*3, 256, 256, 3]
        patches = torch.tensor(patches.transpose(0, 3, 1, 2), device=device)
        if self.int_to_float:
            patches = patches.to(torch.float) / 256
        batch[self.name] = patches

    def __len__(self):
        return len(self.dfcase_high)
    
@register_dataset('tggate_triplet')
class TGGATETripletDataset(Dataset):
    def __init__(self, logger, name, dfs, pdir, n_patch_per_wsi, seed, n_sample,
            int_to_float, split=None):
        super().__init__(logger, name, dfs)
        self.pdir = pdir
        self.n_patch_per_wsi = n_patch_per_wsi
        self.rng = np.random.default_rng(seed)
        self.n_sample = n_sample

        # data
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)
        if split is not None:
            folds = dfs[split['df']][split['col']]
            values = split['values']
            dfcase = dfcase[[fold in values for fold in folds]]

        ## highに限定
        cond2high = {}
        conds = []
        doses = []
        for cond in dfcase['COND']:
            cond = cond.split(';')
            dose = int(cond[4])
            cond = ';'.join(cond[:4])
            
            if cond not in cond2high:
                cond2high[cond] = dose
            else:
                cond2high[cond] = max(cond2high[cond], dose)
            conds.append(cond)
            doses.append(dose)
        dfcase['COND'] = conds
        dfcase['DOSE'] = doses
        is_highest = []
        for cond, dose in zip(conds, doses):
            is_highest.append((cond2high[cond] == dose) and (dose != 0))
        self.dfcase_high = dfcase[is_highest]

        ## controlのcid, high1のcidを集める
        self.cond2cont_cids = {}
        self.cond2high_cids = {}
        for cond in np.unique(self.dfcase_high['COND']):
            dfcase_cond_cont = dfcase[(dfcase['COND'] == cond) & (dfcase['DOSE'] == 0)]
            self.cond2cont_cids[cond] = dfcase_cond_cont.index.values
            dfcase_high_cont = self.dfcase_high[self.dfcase_high['COND'] == cond]
            self.cond2high_cids[cond] = dfcase_high_cont.index.values
        self.int_to_float = int_to_float

    def make_batch(self, batch, idx, device):
        patches = []
        for wsi_idx in idx:
            cid = self.dfcase_high.index[wsi_idx]
            cond = self.dfcase_high['COND'][cid]
            for patch_idx in self.rng.choice(self.n_sample, size=self.n_patch_per_wsi):
                patches.append(np.load(f"{self.pdir}/sample_patch/patch/{cid}/{patch_idx}.npy"))
            ccid = self.rng.choice(self.cond2cont_cids[cond])
            for patch_idx in self.rng.choice(self.n_sample, size=self.n_patch_per_wsi):
                patches.append(np.load(f"{self.pdir}/sample_patch/patch/{ccid}/{patch_idx}.npy"))
        patches = np.stack(patches, axis=0) # [B*2*Np, 256, 256, 3]
        patches = torch.tensor(patches.transpose(0, 3, 1, 2), device=device)
        if self.int_to_float:
            patches = patches.to(torch.float) / 256
        batch[self.name] = patches

    def __len__(self):
        return len(self.dfcase_high)


# 遅かった。
@register_dataset('tggate_triplet_os')
class TGGATETripletOSDataset(Dataset):
    def __init__(self, logger, name, dfs, pdir, n_patch_per_wsi, seed, n_sample,
            int_to_float, path_col, max_workers=None, split=None):
        super().__init__(logger, name, dfs)
        self.pdir = pdir
        self.n_patch_per_wsi = n_patch_per_wsi
        self.rng = np.random.default_rng(seed)
        self.n_sample = n_sample
        self.max_workers = max_workers
        # data
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)
        self.paths = pd.read_csv(f"{pdir}/files2.csv", index_col=0, keep_default_na=False)[path_col]

        ## splitありの場合データを一部抽出
        if split is not None:
            folds = dfs[split['df']][split['col']]
            values = split['values']
            dfcase = dfcase[[fold in values for fold in folds]]


        ## highに限定
        cond2high = {}
        conds = []
        doses = []
        for cond in dfcase['COND']:
            cond = cond.split(';')
            dose = int(cond[4])
            cond = ';'.join(cond[:4])
            
            if cond not in cond2high:
                cond2high[cond] = dose
            else:
                cond2high[cond] = max(cond2high[cond], dose)
            conds.append(cond)
            doses.append(dose)
        dfcase['COND'] = conds
        dfcase['DOSE'] = doses
        is_highest = []
        for cond, dose in zip(conds, doses):
            is_highest.append((cond2high[cond] == dose) and (dose != 0))
        self.dfcase_high = dfcase[is_highest]

        ## controlのcid, high1のcidを集める
        self.cond2cont_cids = {}
        self.cond2high_cids = {}
        for cond in np.unique(self.dfcase_high['COND']):
            dfcase_cond_cont = dfcase[(dfcase['COND'] == cond) & (dfcase['DOSE'] == 0)]
            self.cond2cont_cids[cond] = dfcase_cond_cont.index.values
            dfcase_high_cont = self.dfcase_high[self.dfcase_high['COND'] == cond]
            self.cond2high_cids[cond] = dfcase_high_cont.index.values
        self.int_to_float = int_to_float

        ## patch indexの読み込み
        self.dfcase = dfcase
        self.pindices = {}
        for cid in dfcase.index:
            pindex = []
            for i, file in enumerate(dfcase['FILES'][cid].split('|')):
                pindex0 = np.load(f"{self.pdir}/oindex/{file}.npy")
                pindex0 = np.concatenate([
                    np.full((pindex0.shape[0], 1), fill_value=i, dtype=pindex0.dtype), 
                    pindex0
                ], axis=1)
                pindex.append(pindex0)
            pindex = np.concatenate(pindex)
            self.pindices[cid] = pindex


    def make_batch(self, batch, idx, device):
        patches = []
        for wsi_idx in idx:
            cid = self.dfcase_high.index[wsi_idx]
            
            patches += self.get_patches(cid)

            cond = self.dfcase_high['COND'][cid]
            ccid = self.rng.choice(self.cond2cont_cids[cond])
            # ccid = 348161
            patches += self.get_patches(ccid)
        patches = np.stack(patches, axis=0) # [B*2*Np, 256, 256, 3]
        patches = torch.tensor(patches.transpose(0, 3, 1, 2), device=device)
        if self.int_to_float:
            patches = patches.to(torch.float) / 256
        batch[self.name] = patches

    def get_patches(self, cid):
        pindex = self.rng.choice(self.pindices[cid], size=self.n_patch_per_wsi, 
            replace=False, axis=0)
        if self.max_workers is not None:
            paths = [self.paths[file] for file in self.dfcase['FILES'][cid].split('|')]
            patches = []
            with cf.ProcessPoolExecutor(max_workers=self.max_workers) as e:
                futures = []
                for pindex0 in pindex:
                    iwsi, py, px = list(pindex0)
                    futures.append(e.submit(load_patch, paths[iwsi], py, px))
                for f in futures:
                    patches.append(f.result())
        else:
            wsis = [OpenSlide(self.paths[file]) for file in self.dfcase['FILES'][cid].split('|')]
            patches = []
            for pindex0 in pindex:
                iwsi, py, px = list(pindex0)
                patches.append(np.array(wsis[iwsi].read_region((px*256, py*256), 
                    level=0, size=(256, 256)))[:,:,:3])
        return patches   

    def __len__(self):
        return len(self.dfcase_high)


def load_patch(path, py, px):
    wsi = OpenSlide(path)
    return np.array(wsi.read_region((px*256, py*256), level=0, size=(256, 256)))[:,:,:3]


@register_dataset('tggate_triplet_agg')
class TGGATETripletAggDataset(Dataset):
    def __init__(self, logger, name, dfs, pdir, n_patch_per_wsi, seed, n_sample,
            int_to_float, split=None):
        super().__init__(logger, name, dfs)
        self.pdir = pdir
        self.n_patch_per_wsi = n_patch_per_wsi
        self.rng = np.random.default_rng(seed)
        self.n_sample = n_sample

        # data
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)
        if split is not None:
            folds = dfs[split['df']][split['col']]
            values = split['values']
            dfcase = dfcase[[fold in values for fold in folds]]

        ## highに限定
        cond2high = {}
        conds = []
        doses = []
        for cond in dfcase['COND']:
            cond = cond.split(';')
            dose = int(cond[4])
            cond = ';'.join(cond[:4])
            
            if cond not in cond2high:
                cond2high[cond] = dose
            else:
                cond2high[cond] = max(cond2high[cond], dose)
            conds.append(cond)
            doses.append(dose)
        dfcase['COND'] = conds
        dfcase['DOSE'] = doses
        is_highest = []
        for cond, dose in zip(conds, doses):
            is_highest.append((cond2high[cond] == dose) and (dose != 0))
        self.dfcase_high = dfcase[is_highest]

        ## controlのcid, high1のcidを集める
        self.cond2cont_cids = {}
        self.cond2high_cids = {}
        for cond in np.unique(self.dfcase_high['COND']):
            dfcase_cond_cont = dfcase[(dfcase['COND'] == cond) & (dfcase['DOSE'] == 0)]
            self.cond2cont_cids[cond] = dfcase_cond_cont.index.values
            dfcase_high_cont = self.dfcase_high[self.dfcase_high['COND'] == cond]
            self.cond2high_cids[cond] = dfcase_high_cont.index.values
        self.int_to_float = int_to_float

    def make_batch(self, batch, idx, device):
        patches = []
        bytesize = np.dtype('uint8').itemsize
        for wsi_idx in idx:
            cid = self.dfcase_high.index[wsi_idx]
            for patch_idx in self.rng.choice(self.n_sample, size=self.n_patch_per_wsi):
                patch = np.fromfile(f"{self.pdir}/sample_patch_agg/{cid}.npy", 
                    count=256*256*3, offset=patch_idx*256*256*3*bytesize, 
                    dtype=np.uint8).reshape(256, 256, 3)
                patches.append(patch)
            ccid = self.rng.choice(self.cond2cont_cids[self.dfcase_high['COND'][cid]])
            for patch_idx in self.rng.choice(self.n_sample, size=self.n_patch_per_wsi):
                patch = np.fromfile(f"{self.pdir}/sample_patch_agg/{ccid}.npy", 
                    count=256*256*3, offset=patch_idx*256*256*3*bytesize, 
                    dtype=np.uint8).reshape(256, 256, 3)
                patches.append(patch)
        patches = np.stack(patches, axis=0) # [B*2*Np, 256, 256, 3]
        patches = torch.tensor(patches.transpose(0, 3, 1, 2), device=device)
        if self.int_to_float:
            patches = patches.to(torch.float) / 256
        batch[self.name] = patches

    def __len__(self):
        return len(self.dfcase_high)
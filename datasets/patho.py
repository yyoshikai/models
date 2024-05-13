import numpy as np
import pandas as pd
import torch

from ..dataset import Dataset, register_dataset

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
            int_to_float):
        super().__init__(logger, name, dfs)
        self.pdir = pdir
        self.n_patch_per_wsi = n_patch_per_wsi
        self.rng = np.random.default_rng(seed)
        self.n_sample = n_sample

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

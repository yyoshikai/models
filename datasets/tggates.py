import numpy as np
import pandas as pd
from ..dataset import Dataset

# 240219作成
# かなりspecific.
# TGGATEsについて, 対照群と処置群のペアを出力する。
info_kwargs = dict(header=[0,1], sep='\t', keep_default_na=False)
class TGGATEControlDataset(Dataset):
    def __init__(self, logger, name, dfs, 
            info_path, wsi_path, pindex_path, ppi, seed=0, 
            patch_size=256):
        """

        Info(csv)の情報
        ---------------
        n_patch: サンプルしたpatch数
        control_file: コントロールのファイル名。
        
        
        """
        super().__init__(logger, name, dfs)
        self.info = pd.read_csv(info_path, **info_kwargs)
        self.ppi = ppi
        self.wsi_path = wsi_path
        self.pindex_path = pindex_path
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size

        # modules
        from openslide import OpenSlide
        self.open_class = OpenSlide
    
    def make_batch(self, batch, idx, device=None):
        patches_c = []
        patches_t = []
        for i in idx:
            files = self.info['FILES', 'Liver'][i]
            period = self.info['PERIOD', '-'][i].replace(' ', '_')
            file = self.rng.choice(files.split('|'))
            wsi = self.open_class(f"{self.wsi_path}/{period}/{file}")
            pindices = np.load(f"{self.pindex_path}/{file}.npy")
            pindices = self.rng.choice(pindices, size=self.ppi, replace=False)
            for pindex in pindices:
                patch = wsi.read_region((pindex[1]*self.patch_size, pindex[0]*self.patch_size),
                    size=(self.patch_size, self.patch_size), level=0)
                patches_t.append(patch)                                                                                                                                    
    
    def __len__(self):
        return len(self.info)

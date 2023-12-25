import numpy as np
import pandas as pd
from ..dataset import DataLoader

# かなりspecific.
# TGGATEsについて, 対照群と処置群のペアを出力する。
class TGGATEControlDataloader(DataLoader):
    def __init__(self, logger, seed, device,
            info_path, batch_size, checkpoint=None, **kwargs):
        """

        Info(csv)の情報
        ---------------
        n_patch: サンプルしたpatch数
        control_file: コントロールのファイル名。
        
        
        """
        self.logger = logger
        self.info = pd.read_csv(info_path)
        self.accum_patch = np.cumsum(self.info['n_patch']).astype(int)
        self.i_cur_image = 0
        self.i_cur_patch = 0
        self.batch_size = batch_size



    def checkpoint(self, path_checkpoint):
        raise NotImplementedError
    def get_batch(self, batch=None):
        if batch is None: batch = {}







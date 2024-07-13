"""
LatentTransformerのfeat_evalとgenerationを統合
"""
import sys, os
os.environ.setdefault('TOOLS_DIR', "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
import yaml
import pickle
from addict import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tools.notice import noticeerror
from tools.path import make_result_dir
from tools.logger import default_logger
from tools.args import load_config2
from tools.notice import notice
from models.models2 import Model
from models.process import get_process
from models.accumulator import get_accumulator, NumpyAccumulator, ListAccumulator
from models.metric import get_metric
from models.dataset import get_dataloader

@noticeerror("models/evaluate", notice_end=False)
def main(config):
    """
    result_dir:
        
    study_dir: str
    step: int
    logger:
        Input for default_logger
        stream_level: [Optional]str
        file_level: [Optional]str
    data: dict
        input for get_dataloader
    gpuid: int
    model: [Optional] Modification of model config
    replace: [Optional]dict
    strict: [Optional]bool
    processes: list
    show_tqdm: bool
    notice: bool

    metrics
    accumulators
    
    """

    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir+"/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    # Environment
    DEVICE = torch.device('cuda', index=config.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    
    # Prepare data
    dl = get_dataloader(logger=logger, device=DEVICE, **config.data)

    # Prepare model
    logger.info("Preparing model...")
    with open(f"{config.study_dir}/config.yaml") as f:
        study_config = Dict(yaml.load(f, yaml.Loader))
    model_config = study_config.model
    model_config.update(config.model)
    model = Model(logger, **model_config)
    model.load(path=f"{config.study_dir}/models/{config.step}", 
        replace=config.replace, strict=bool(config.strict))
    model.to(DEVICE)
    model.eval()
    processes = [get_process(**p) for p in config.processes]

    # Prepare hooks
    accums = {aname: get_accumulator(logger=logger, **aconfig)
        for aname, aconfig in config.accumulators.items()}
    idx_accum = NumpyAccumulator(logger=logger, input='idx', org_type='np.ndarray')
    metrics = [get_metric(logger=logger, name=mname, **mconfig) for mname, mconfig
        in config.metrics.items()]
    hooks = list(accums.values())+metrics+[idx_accum]

    for hook in hooks:
        hook.init()

    # Iteration
    logger.info("Iterating dataset...")
    with torch.no_grad():
        for batch in tqdm(dl) if config.show_tqdm else dl:
            model(batch, processes=processes)
            for hook in hooks:
                hook(batch)
            del batch
            torch.cuda.empty_cache()

    # Calculate metrics
    logger.info("Calculating metrics...")
    if len(metrics) > 0:
        scores = {}
        for m in metrics: 
            scores = m.calc(scores)
        df_score = pd.Series(scores)
        df_score.to_csv(f"{result_dir}/scores.csv", header=['Score'])
    
    # Save accumulated values
    logger.info("Saving accumulates...")
    if len(accums) > 0:
        idxs = np.argsort(idx_accum.accumulate())
        for aname, accum in accums.items():
            accummed = accum.accumulate()
            apath = f"{result_dir}/{aname}"
            if isinstance(accum, NumpyAccumulator):
                accummed = accummed[idxs]
                n, size = accummed.shape
                save_type = config.save.get(aname, 'csv')
                if save_type == 'csv':
                    with open(apath+'.csv', 'w') as f:
                        f.write(','.join([str(i) for i in range(size)])+'\n')
                        for r in range(n):
                            f.write(','.join(str(f) for f in accummed[r])+'\n')
                elif save_type == 'npy':
                    np.save(apath+'.npy', accummed)
                else:
                    raise ValueError(f"Unsupported save_type: {save_type}")
            elif isinstance(accum, ListAccumulator):
                accummed = [accummed[i] for i in idxs]
                with open(apath+'.pkl', 'wb') as f:
                    pickle.dump(accummed, f)
            else:
                raise ValueError(f"Unsupported type of accumulate: {type(accum)}")
    if config.notice:
        notice(f"From feat_eval_gen: finished!")


if __name__ == '__main__':
    default_configs = []
    if os.path.exists('base.yaml'):
        default_configs.append('base')
    config = load_config2(config_dir="", default_configs=default_configs)
    main(config)


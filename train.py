import sys, os
"""
・共通のModelを使うように変更
・train, valは必ず複数セット

240402 optimizer.zero_grad()をtrain開始時からoptimizer作成時に移動
    opt_freq >= 2でload_state_dictする場合結果が変わると思ったので。
"""

os.environ.setdefault("TOOLS_DIR", "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
import pickle
import time
import random
import shutil
from copy import deepcopy
from contextlib import nullcontext

import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tools.notice import notice, noticeerror
from tools.path import make_result_dir, timestamp
from tools.logger import default_logger
from tools.args import load_config2, subs_vars
from tools.torch import get_params
from models.dataset import get_dataloader
from models.accumulator import get_accumulator, NumpyAccumulator
from models.metric import get_metric
from models.optimizer import get_scheduler
from models.process import get_processes
from models.hooks import AlarmHook, hook_type2class, get_hook
from models import Model
from models.optimizer import ModelOptimizer
from models.utils import get_set

def save_rstate(dirname):
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/random.pkl", 'wb') as f:
        pickle.dump(random.getstate(), f)
    with open(f"{dirname}/numpy.pkl", 'wb') as f:
        pickle.dump(np.random.get_state(), f)
    torch.save(torch.get_rng_state(), f"{dirname}/torch.pt")
    torch.save(torch.cuda.get_rng_state_all(), f"{dirname}/cuda.pt")

def set_rstate(config):
    if 'random' in config:
        with open(config.random, 'rb') as f:
            random.setstate(pickle.load(f))
    if 'numpy' in config:
        with open(config.numpy, 'rb') as f: 
            np.random.set_state(pickle.load(f))
    if 'torch' in config:
        torch.set_rng_state(torch.load(config.torch))
    if 'cuda' in config:
        torch.cuda.set_rng_state_all(torch.load(config.cuda))

@noticeerror(from_=f"train.py in {os.getcwd()}", notice_end=False)
def main(config, args=None):

    # make training
    trconfig = config.training
    result_dir = make_result_dir(**trconfig.result_dir)
    logger = default_logger(result_dir+"/log.txt", trconfig.verbose.loglevel.stream or 'info', trconfig.verbose.loglevel.file or 'debug')
    with open(result_dir+"/config.yaml", mode='w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)
    logger.warning(f"options: {' '.join(args)}")

    # environment
    ## device
    DEVICE = torch.device('cuda', index=trconfig.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    ## detect anomaly
    if trconfig.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    ## deterministic
    if trconfig.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False
    
    # prepare data
    dl_train = get_dataloader(logger=logger, device=DEVICE, **trconfig.data.train)
    dls_val = {name: get_dataloader(logger=logger, device=DEVICE, **dl_val_config)
        for name, dl_val_config in trconfig.data.vals.items()}
    n_epoch = trconfig.n_epoch or 1
    
    # prepare model
    if 'model_seed' in trconfig:
        random.seed(trconfig.model_seed)
        np.random.seed(trconfig.model_seed)
        torch.manual_seed(trconfig.model_seed)
        torch.cuda.manual_seed(trconfig.model_seed)
    model = Model(logger=logger, **config.model)
    if trconfig.init_weight:
        model.load(**trconfig.init_weight)
    df_param, n_param, bit_size = get_params(model)
    df_param.to_csv(f"{result_dir}/parameters.tsv", sep='\t', index=False)
    logger.info(f"# of params: {n_param}")
    logger.info(f"Model size(bit): {bit_size}")
    model.to(DEVICE)

    # prepare optimizer
    optimizers = {
        oname: ModelOptimizer(name=oname, model=model, dl_train=dl_train, n_epoch=n_epoch, **oconfig)
        for oname, oconfig in optimizers.items()
    }
    
    # process
    train_processes = get_processes(trconfig.train_loop)
    val_processes = get_processes(trconfig.val_loop, 
            trconfig.train_loop if trconfig.val_loop_add_train else None)

    class SchedulerAlarmHook(AlarmHook):
        def __init__(self, scheduler, optimizer='loss', **kwargs):
            super().__init__(**kwargs)
            scheduler.setdefault('last_epoch', dl_train.step - 1)
            self.scheduler = get_scheduler(optimizers[optimizer].optimizer, 
                dl_train=dl_train, n_epoch=n_epoch, opt_freq=trconfig.schedule.opt_freq,
                **scheduler)
        def ring(self, batch, model):
            self.scheduler.step()
    hook_type2class['scheduler_alarm'] = SchedulerAlarmHook

    # Prepare abortion
    abortion: dict = trconfig.abortion
    abort_time = abortion.pop('time', None)
    minus_abortion: dict = trconfig.mabortion

    # Prepare notice
    notice_studyname = trconfig.notice.pop('studyname', "")
    notice_end = bool(trconfig.notice.end)
    notice_points = {key: get_set(value) for key, value in trconfig.notice.points}

    # Prepare metrics
    accumulators = [ get_accumulator(**acc_config) for acc_config in trconfig.accumulators ]
    idx_accumulator = NumpyAccumulator(input='idx')
    metrics = [ get_metric(name=name, **met_config) for name, met_config in trconfig.metrics.items() ]
    scores_df = pd.read_csv(trconfig.stocks.score_df, index_col="Step") \
        if trconfig.stocks.score_df else  pd.DataFrame(columns=[], dtype=float)
    class ValidationAlarmHook(AlarmHook):
        def __init__(self, eval_train = False, **kwargs):
            super().__init__(**kwargs)
            self.eval_train = eval_train

        eval_steps = []
        def ring(self, batch, model):
            step = batch['step']
            if step in self.eval_steps: return
            self.logger.info(f"Validating step{step:7} ...")
            if not self.eval_train:
                model.eval()
            with torch.no_grad():
                for key, dl in dls_val.items():
                    dlname = "" if key == "base" else f"_{key}"
                    ## evaluation
                    for x in metrics+accumulators+[idx_accumulator]: x.init()
                    for batch0 in dl:
                        batch0 = model(batch0, processes=val_processes)
                        for x in metrics+accumulators+[idx_accumulator]:
                            x(batch0)
                        del batch0
                        torch.cuda.empty_cache()
                    
                    ## calculate & save score
                    scores = {}
                    for metric in metrics: scores = metric.calc(scores)
                    batch.update({key+dlname: value for key, value in scores.items()})
                    for score_lb, score in scores.items():
                        self.logger.info(f"  {score_lb:20}: {score:.3f}")
                        scores_df.loc[step, score_lb] = score
                    scores_df.to_csv(f"{result_dir}/val_score{dlname}.csv", index_label="Step")

                    ## save accumulated
                    idx = idx_accumulator.accumulate()
                    idx = np.argsort(idx)
                    for accumulator in accumulators:
                        accumulator.save(f"{result_dir}/accumulates/{accumulator.input}{dlname}/{step}", indices=idx)
            
            ## save steps
            self.eval_steps.append(step)
            model.train()
    hook_type2class['validation_alarm'] = ValidationAlarmHook

    # prepare checkpoint
    ## alarmが複数の場合を考えるとcheckpoint_stepsは共通のほうがよい?
    class CheckpointAlarm(AlarmHook):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)
            self.checkpoint_steps = []
        def ring(self, batch, model: Model):
            if batch['step'] in self.checkpoint_steps: return
            checkpoint_dir = f"{result_dir}/checkpoints/{batch['step']}"
            logger.info(f"Making checkpoint at step {batch['step']:6>}...")
            if len(self.checkpoint_steps) > 0:
                shutil.rmtree(f"{result_dir}/checkpoints/{self.checkpoint_steps[-1]}/")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_state_dict(f"{result_dir}/models/{batch['step']}")
            for optimizer in optimizers.values():
                path = f"{checkpoint_dir}/{optimizer.filename}.pth"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(optimizer.optimizer.state_dict(), path)
            dl_train.checkpoint(f"{checkpoint_dir}/dataloader_train")
            scores_df.to_csv(checkpoint_dir+"/val_score.csv", index_label="Step")
            save_rstate(f"{checkpoint_dir}/rstate")
            self.checkpoint_steps.append(batch['step'])
    hook_type2class['checkpoint_alarm'] = CheckpointAlarm
    pre_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.pre_hooks.values()]
    post_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.post_hooks.values()]

    # log config
    lconfigs = []
    lconfig = Dict()
    for lconfig0 in trconfig.loop_logs:
        lconfig.update(lconfig0)
        lconfigs.append(deepcopy(lconfig))
    max_loop_log_level = max([lc.level for lc in lconfigs]) \
        if len(lconfigs) > 0 else 0
        
    # load random state
    set_rstate(trconfig.rstate)

    # training
    training_start = time.time()
    logger.info("Training started.")
    with (tqdm(total=n_epoch*(dl_train.get_len() or 0), initial=dl_train.step) if trconfig.verbose.show_tqdm else nullcontext()) as pbar:
        
        while True:
            now = time.time()
            # pre hooks
            batch = {'step': dl_train.step, 'epoch': dl_train.epoch}
            for hook in pre_hooks:
                hook(batch, model)
            
            # abortion
            for key, value in abortion.items():
                if key in batch and batch[key] >= value:
                    batch['end'] = True
            for key, value in minus_abortion.items():
                if key in batch and batch[key] <= value:
                    batch['end'] = True
            if abort_time is not None and now - training_start >= abort_time:
                batch['end'] = True
            if 'end' in batch: break

            # training
            batch = dl_train.get_batch(batch)
            if len(lconfigs) > 0:
                logger.log(level=max_loop_log_level, msg=f"Step {dl_train.step}: ")

            # Log shapes
            for lconfig in lconfigs:
                item = batch[lconfig.name]
                msg = f"  {lconfig.name}: "
                if lconfig.type == 'value':
                    msg += str(item)
                elif lconfig.type == 'shape':
                    msg += str(list(item.shape))
                logger.log(level=lconfig.level, msg=msg)

            start = time.time()
            batch = model(batch, processes=train_processes, logger=logger)
            
            for optimizer0 in optimizers.values():
                optimizer0.step(batch)
            
            batch['time'] = time.time() - start
                
            for hook in post_hooks:
                hook(batch, model)
            del batch
            torch.cuda.empty_cache()
            if pbar is not None: pbar.update(1)

            # notice
            for key, values in notice_points.items():
                if batch[key] in values:
                    notice(f"models/train.py {notice_studyname} {key} {batch[key]} finished!")
    if notice_end:
        notice(f"models/train.py: {notice_studyname} finished!")
    for hook in pre_hooks+post_hooks:
        hook(batch, model)

if __name__ == '__main__':
    config = load_config2("", default_configs=['base.yaml'])
    ## replacement of config: add more when needed
    config = subs_vars(config, {"$TIMESTAMP": timestamp()})
    main(config, sys.argv)

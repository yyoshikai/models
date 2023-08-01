import sys, os

PROJECT_DIR = os.environ['PROJECT_DIR'] if "PROJECT_DIR" in os.environ \
    else "/workspace/LatentTransformer"
RESULT_DIR = PROJECT_DIR+"/results/training"
DATA_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ.keys() \
    else PROJECT_DIR+"/data"
os.environ.setdefault("TOOLS_DIR", "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
import pickle
import time
import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

from tools.notice import notice, noticeerror
from tools.path import make_result_dir, timestamp
from tools.logger import default_logger
from tools.args import load_config2, subs_vars
from models.dataset import get_dataloader
from models.accumulator import get_accumulator
from models.metric import get_metric
from models.alarm import get_alarm, SilentAlarm
from models.optimizer import get_optimizer, get_scheduler
from tools.tools import nullcontext
from src.version import VERSION
from models import Model
torch.autograd.set_detect_anomaly(True)

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

@noticeerror(from_="LatentTransformer/training", notice_end=False)
def main(config, args=None):

    # make training
    ## replacement of config: add more when needed
    env_config = {
        "$VERSION": f"{VERSION:02}",
        "$PROJECT_DIR": PROJECT_DIR,
        "$DATA_DIR": DATA_DIR,
        "$RESULT_DIR": RESULT_DIR,
        "$TIMESTAMP": timestamp(),
    }
    config = subs_vars(config, env_config)
    trconfig = config.training
    ## Make result directory
    result_dir = make_result_dir(trconfig.result_dir.dirname,
        duplicate=trconfig.result_dir.duplicate)
    os.makedirs(result_dir+"/models", exist_ok=True)
    os.makedirs(result_dir+"/checkpoints/cache", exist_ok=True)
    ## Make logger
    logger = default_logger(result_dir+"/log.txt", trconfig.verbose.loglevel.stream,
        trconfig.verbose.loglevel.file)
    ## save training conditions
    with open(result_dir+"/config.yaml", mode='w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)
    shutil.copy2(__file__, result_dir+"/training.py")
    if args is not None:
        logger.warning(f"options: {' '.join(args)}")

    # prepare data
    ## device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.warning(f"DEVICE: {DEVICE}")
    ## dataset
    sizes = {}
    dl_train = get_dataloader(logger=logger, device=DEVICE, sizes=sizes, **trconfig.data.train)
    dls_val = {name: get_dataloader(logger=logger, device=DEVICE, sizes={}, **dl_val_config)
        for name, dl_val_config in trconfig.data.vals.items()}
    
    # prepare model
    model = Model(logger=logger, config=config.model, seed=trconfig.init_seed)
    if trconfig.init_weight:
        model.load_state_dict(**trconfig.init_weight)
    model.to(DEVICE)
    optimizer = get_optimizer(type=trconfig.optimizer.type, params=model.parameters(), **trconfig.optimizer.args)
    if trconfig.optimizer.init_weight:
        optimizer.load_state_dict(torch.load(trconfig.optimizer.init_weight))
    scheduler = get_scheduler(optimizer, **trconfig.scheduler, 
        last_epoch=trconfig.schedule.scheduler.last_epoch or dl_train.step - 1)
    
    # Prepare alarms
    abort_step = trconfig.abortion.step or float('inf')
    abort_epoch = trconfig.abortion.epoch or float('inf')
    abort_time = trconfig.abortion.time or float('inf')
    alarms = Dict()
    for alarm_type in ['validation', 'loss_manager', 'notice', 'save_model', 'checkpoint', 'scheduler']:
        for t_type in ['step', 'epoch']:
            if trconfig.schedule[alarm_type][t_type]:
                alarms[alarm_type][t_type] = get_alarm(logger=logger, **trconfig.schedule[alarm_type][t_type])
            else:
                alarms[alarm_type][t_type] = SilentAlarm(logger=logger)

    # Prepare metrics
    accumulators = [ get_accumulator(logger=logger, **acc_config) for acc_config in trconfig.accumulators ]
    metrics = [ get_metric(logger=logger, name=name, **met_config) for name, met_config in trconfig.metrics.items() ]
    if trconfig.stocks.score_df:
        scores_df = pd.read_csv(trconfig.stocks.score_df, index_col="Step")
    else:
        scores_df = pd.DataFrame(columns=[], dtype=float)
    dfs_steps = []
    if trconfig.stocks.steps_df:
        dfs_steps.append(pd.read_csv(trconfig.stocks.steps_df))
    list_steps = defaultdict(list)
    eval_steps = []
    def eval_():
        logger.info(f"Validating step{dl_train.step:7} ...")
        model.eval()
        ## evaluation
        for x in metrics + accumulators: x.init()
        with torch.no_grad():
            for key, dl in dls_val.items():
                for metric in metrics:
                    metric.set_val_name(key)
                for batch in dl:
                    model(batch, mode='evaluate')
                    for x in metrics+accumulators:
                        x(batch)
                    del batch
                    torch.cuda.empty_cache()
                
        ## calculate & save score
        scores = {}
        for metric in metrics: scores = metric.calc(scores)
        for score_lb, score in scores.items():
            logger.info(f"  {score_lb:20}: {score:.3f}")
            scores_df.loc[dl_train.step, score_lb] = score
        scores_df.to_csv(result_dir+"/val_score.csv", index_label="Step")

        ## save accumulated
        for accumulator in accumulators:
            accumulator.save(f"{result_dir}/accumulates/{accumulator.input}/{dl_train.step}")
        
        ## save steps
        dfs_steps.append(pd.DataFrame(list_steps))
        list_steps.clear()
        pd.concat(dfs_steps).to_csv(result_dir+"/steps.csv", index=False)
        eval_steps.append(dl_train.step)
        model.train()
        return scores

    # prepare checkpoint
    checkpoint_steps = []
    def make_checkpoint():
        logger.info(f"Making checkpoint at step {dl_train.step:6>}...")
        if len(checkpoint_steps) > 0:
            former_checkpoint_step = checkpoint_steps[-1]
            if os.path.exists(f"{result_dir}/checkpoints/{former_checkpoint_step}/.model_saved"):
                os.remove(f"{result_dir}/models/{former_checkpoint_step}.pth")
            shutil.rmtree(f"{result_dir}/checkpoints/{former_checkpoint_step}/")
        os.makedirs(f"{result_dir}/checkpoints/{dl_train.step}", exist_ok=True)
        if not os.path.exists(f"{result_dir}/models/{dl_train.step}.pth"):
            torch.save(model.state_dict(), f"{result_dir}/models/{dl_train.step}.pth")
            open(f"{result_dir}/checkpoints/{dl_train.step}/.model_saved", 'w').close()            
        torch.save(optimizer.state_dict(), f"{result_dir}/checkpoints/{dl_train.step}/optimizer.pth")
        dl_train.checkpoint(f"{result_dir}/checkpoints/{dl_train.step}/dataloader_train")
        save_rstate(f"{result_dir}/checkpoints/{dl_train.step}/rstate")
        dfs_steps.append(pd.DataFrame(list_steps))
        list_steps.clear()
        scores_df.to_csv(f"{result_dir}/checkpoints/{dl_train.step}/val_score.csv", index_label="Step")
        pd.concat(dfs_steps).to_csv(f"{result_dir}/checkpoints/{dl_train.step}/steps.csv", index=False)
        checkpoint_steps.append(dl_train.step)

    # load & save random state
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_rstate(trconfig.rstate)

    # training
    training_start = time.time()
    logger.info("Training started.")
    max_step = abort_step
    
    with (tqdm(total=max_step if max_step < float('inf') else None,
            desc=None, initial=dl_train.step) if trconfig.verbose.show_tqdm else nullcontext()) as pbar:
        now = time.time()
        optimizer.zero_grad()
        while True:
            # validation
            if alarms.validation.step(dl_train.step) | alarms.validation.epoch(dl_train.epoch):
                if len(list_steps['Time']) > 0:
                    logger.info(f"  Time per step: {np.mean(list_steps['Time']):3.3f}")
                eval_()
            if alarms.save_model.step(dl_train.step) | alarms.save_model.epoch(dl_train.epoch):
                torch.save(model.state_dict(), f"{result_dir}/models/{dl_train.step}.pth")
            if alarms.notice.step(dl_train.step):
                notice(f"From LatentTF: Study {trconfig.studyname} {dl_train.step:>6} step finished.")
            if alarms.notice.epoch(dl_train.epoch):
                notice(f"From LatentTF: Study {trconfig.studyname} {dl_train.epoch:>4} epoch finished.")
            if alarms.checkpoint.step(dl_train.step) or alarms.checkpoint.epoch(dl_train.epoch):
                make_checkpoint()
            if dl_train.step >= abort_step or now - training_start >= abort_time \
                or dl_train.epoch >= abort_epoch: break

            # training
            batch = dl_train.get_batch()
            batch['step'] = dl_train.step
            batch['loss'] = 0
            start = time.time()
            batch = model(batch, mode='train')
            torch.cuda.empty_cache()
            loss = batch['loss']
            loss.backward()
            if trconfig.regularize_loss.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                max_norm=trconfig.regularize_loss.clip_grad_norm, error_if_nonfinite=True)
            if dl_train.step % trconfig.schedule.opt_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            if alarms.scheduler.step(dl_train.step) or alarms.scheduler.epoch(dl_train.epoch):
                scheduler.step()
            
            # record step-wise scores
            list_steps['Step'].append(dl_train.step)
            list_steps['Loss'].append(loss.detach().cpu().numpy())
            now = time.time()
            list_steps['Time'].append(now-start)
            del batch, loss
            torch.cuda.empty_cache()
            if pbar is not None:
                pbar.update(1)
    if trconfig.schedule.save_model.end and not os.path.exists(f"{result_dir}/models/{dl_train.step}.pth"):
        torch.save(model.state_dict(), f"{result_dir}/models/{dl_train.step}.pth")
    if trconfig.schedule.validation.end and dl_train.step not in eval_steps:
        logger.info(f"  Time per step: {np.mean(list_steps['Time']):3.3f}")
        eval_()
    if trconfig.schedule.checkpoint.end and dl_train.step not in checkpoint_steps:
        make_checkpoint()
    if trconfig.schedule.notice.end:
        notice(f"From LatentTF: {trconfig.studyname} finished!")

if __name__ == '__main__':
    config = load_config2("configs/training04", default_configs=[])
    main(config, sys.argv)

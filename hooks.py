import os
from collections import defaultdict
import pandas as pd
from .alarm import get_alarm


class AlarmHook:
    def __init__(self, logger, result_dir, target, alarm, end=False):
        self.logger = logger
        self.target = target
        self.alarm = get_alarm(logger=logger, **alarm)
        self.end = end
    def __call__(self, batch, model):
        if self.alarm(batch[self.target]) or \
            ('end' in batch and self.end):
            self.ring(batch=batch, model=model)
    def ring(self, batch, model):
        raise NotImplementedError
class SaveAlarmHook(AlarmHook):
    def __init__(self, logger, result_dir, target, alarm, end=False):
        super().__init__(logger, result_dir, target, alarm, end=False)
        self.models_dir = f"{result_dir}/models"
        os.makedirs(self.models_dir, exist_ok=True)
    def ring(self, batch, model):
        self.logger.info(f"Saving model at step {batch['step']:2>}...")
        path = f"{self.models_dir}/{batch['step']}"
        if not os.path.exists(path):
            model.save_state_dict(path)
hook_type2class = {
    'save_alarm': SaveAlarmHook
}
class AccumulateHook:
    def __init__(self, logger, result_dir, names, cols, save_alarm, checkpoint=None):
        os.makedirs(result_dir, exist_ok=True)
        self.path_df = result_dir+"/steps.csv"
        self.save_alarm = get_alarm(logger, **save_alarm)
        self.dfs = []
        if checkpoint is not None:
            self.dfs.append(pd.read_csv(checkpoint))
        self.lists = defaultdict(list)
        self.names = names
        self.cols = cols

    def __call__(self, batch, model):
        for name, col in zip(self.names, self.cols):
            self.lists[col].append(batch[name])        
        if ():
            self.dfs.append(pd.DataFrame(self.lists))
            self.lists.clear()
            pd.concat(self.dfs).to_csv(self.path_df, index=False)

            if len(list_steps['Time']) > 0:
                self.logger.info(f"  Time per step: {np.mean(list_steps['Time']):3.3f}")
            

def get_hook(type, **kwargs):
    return hook_type2class[type](**kwargs)
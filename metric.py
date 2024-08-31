"""
240723 val_nameを削除
"""

from collections import defaultdict
import itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, \
    mean_squared_error, mean_absolute_error, r2_score

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
    def init(self):
        raise NotImplementedError
    def calc(self, scores: dict):
        raise NotImplementedError
    def __call__(self, batch):
        raise NotImplementedError

class BinaryMetric(Metric):
    def __init__(self, name, input, target, is_logit=None, is_multitask=False, 
            input_process=None, task_names=None, mean_multitask=False, **kwargs):
        """
        Parameters
        ----------
        is_logit: bool
            If True, batch[input] is used as decision function.
            If False, batch[input][:, 1] is used as decision function. 
        is_multitask: bool
            If True, decision function should be [batch_size, n_task(, 2)]
            If False, decision function should be [batch_size(, 2)]
        input_process: str or None
            None: nothing applied to decision function
            'softmax': softmax function is applied (is_logit must be False)
            'sigmoid': sigmoid function is applied (is_logit must be True)
        task_names: List[str] or None
        """
        super().__init__(name, **kwargs)
        self.input = input
        self.target = target
        self.is_logit = is_logit
        assert input_process is None or input_process in {'softmax', 'sigmoid'}
        if self.is_logit is not None:
            self._check_input_process()
        self.is_multitask = is_multitask
        self.input_process = input_process
        self.task_names = task_names
        self.mean_multitask = mean_multitask
    def _check_input_process(self):
            if self.input_process == 'softmax': assert not self.is_logit
            elif self.input_process == 'sigmoid': assert self.is_logit

    def init(self):
        self.targets = []
        self.inputs = []
    def __call__(self, batch):
        self.targets.append(batch[self.target].cpu().numpy())
        input = batch[self.input]
        if self.is_logit is None:
            if self.is_multitask:
                self.is_logit = input.dim() == 2
            else:
                self.is_logit = input.dim() == 1
            self._check_input_process()
            if not self.is_logit:
                assert input.size()[-1] == 2
        if self.input_process == 'softmax':
            input = torch.softmax(input, dim=-1)
        elif self.input_process == 'sigmoid':
            input = torch.sigmoid(input)
        input = input.cpu().numpy()
        if not self.is_logit:
            input = input[..., 1]
        self.inputs.append(input)
    def calc(self, scores):

        input = np.concatenate(self.inputs)
        target = np.concatenate(self.targets)
        if len(input) == 0: return scores
        if self.is_multitask:
            if self.mean_multitask:
                scores0 = [ self.calc_score(y_true=target[:, i_task], y_score=input[:, i_task])
                        for i_task in range(target.shape[1])]
                scores[self.name] = np.mean(scores0)
            else:
                for i_task, task_name in zip(range(target.shape[1]), self.task_names):
                    scores[f"{self.name}_{task_name}"] = \
                        self.calc_score(y_true=target[:,i_task], y_score=input[:, i_task])
        else:
            scores[self.name] = self.calc_score(y_true=target, y_score=input)
        return scores
    def calc_score(self, y_true, y_score):
        raise NotImplementedError
        
class AUROCMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        if np.all(y_true == y_true[0]):
            return 0
        else:
            return roc_auc_score(y_true=y_true, y_score=y_score)
class AUPRMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        if np.all(y_true == y_true[0]):
            return 0
        else:
            return average_precision_score(y_true=y_true, y_score=y_score)
class RMSEMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_score))
class MAEMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return mean_absolute_error(y_true=y_true, y_pred=y_score)
class R2Metric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return r2_score(y_true=y_true, y_pred=y_score)
    
class GANAUROCMetric(Metric):
    def __init__(self, name, real, fake):
        super().__init__(name)
        self.real_name = real
        self.fake_name = fake
    def init(self):
        self.real_preds = []
        self.fake_preds = []
    def __call__(self, batch):
        self.real_preds.append(batch[self.real_name].cpu().numpy())
        self.fake_preds.append(batch[self.fake_name].cpu().numpy())
    def calc(self, scores):
        reals = np.concatenate(self.real_preds)
        fakes = np.concatenate(self.fake_preds)
        real_targets = np.ones_like(reals, dtype=int)
        fake_targets = np.zeros_like(fakes, dtype=int)
        score = roc_auc_score(
            np.concatenate([real_targets, fake_targets]),
            np.concatenate([reals, fakes])
        )
        scores[self.name] = score
        return scores

class MeanMetric(Metric):
    def init(self):
        self.values = []
    def calc(self, scores):
        if self.values[0].ndim == 0:
            values = np.array(self.values)
        else:
            values = np.concatenate(self.values)
        if len(values) > 0:
            scores[self.name] = np.mean(values)
        return scores
class ValueMetric(MeanMetric):
    def __call__(self, batch):
        self.values.append(batch[self.name].cpu().numpy())
class PerfectAccuracyMetric(MeanMetric):
    def __init__(self, name, input, target, pad_token, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.input = input
        self.target = target
        self.pad_token = pad_token
    def __call__(self, batch):
        self.values.append(torch.all((batch[self.input] == batch[self.target])
            ^(batch[self.target] == self.pad_token), axis=1).cpu().numpy())
class PartialAccuracyMetric(MeanMetric):
    def __init__(self, name, input, target, pad_token, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.input = input
        self.target = target
        self.pad_token = pad_token
    def __call__(self, batch):
        target_seq = batch[self.target]
        pred_seq = batch[self.input]
        pad_mask = (target_seq != self.pad_token).to(torch.int)
        self.values.append((torch.sum((target_seq == pred_seq)*pad_mask, dim=1)
            /torch.sum(pad_mask, dim=1)).cpu().numpy())
metric_type2class = {
    'value': ValueMetric,
    'auroc': AUROCMetric,
    'aupr': AUPRMetric,
    'rmse': RMSEMetric,
    'mae': MAEMetric,
    'r2': R2Metric,
    'gan_auroc': GANAUROCMetric,
    'perfect': PerfectAccuracyMetric,
    'partial': PartialAccuracyMetric,
}
def get_metric(name, type, **kwargs) -> Metric:
    return metric_type2class[type](name=name, **kwargs)
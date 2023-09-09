# old version
import logging
from functools import partial
import torch
import torch.nn as nn
from ..utils import check_leftargs, EMPTY

class CECriterion(nn.Module):
    def __init__(self, logger: logging.Logger, 
            sizes: dict, input: dict = {'pred': 'pred', 'target': 'target'},
            output: str='CELoss', reduction: str='mean', ignore_index: int=-100, **kwargs):
        """
        batch[input['pred']]: [*, n_class]
        batch[input['target']]: [*]
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input_pred = input['pred']
        self.input_target = input['target']
        self.n_class = sizes[self.input_pred][-1]
        self.output = output
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
        sizes[self.output] = []
    def forward(self, batch: dict, mode: str):
        batch[self.output] = self.criterion(input=batch[self.input_pred].view(-1, self.n_class),
            target=batch[self.input_target].ravel())

class BCEWithLogitsCriterion(nn.Module):
    def __init__(self, logger: logging.Logger, 
            sizes: dict, input: dict = {'pred': 'pred', 'target': 'target'},
            output: str='CELoss', reduction: str='mean', **kwargs):
        """
        Batch
        -----
        batch[input['pred]]: torch.tensor(float)[*]
        batch[target['pred]]: torch.tensor(long)[*]
        batch[output]: float if reduction == 'mean' or 'sum' else torch.tensor(float)[*]
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input_pred = input['pred']
        self.input_target = input['target']
        self.output = output
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        sizes[self.output] = sizes[self.input_pred] if reduction == 'none' \
            else []
    def forward(self, batch: dict, mode: str):
        batch[self.output] = self.criterion(input=batch[self.input_pred], target=batch[self.input_target])
        return batch

class MSECriterion(nn.Module):
    def __init__(self, logger: logging.Logger, sizes: dict, input: dict = {'pred': None, 'target': None},
        output: str='MSELoss', reduction: str='mean', normalize: bool=False, **kwargs):
        """
        Batch
        -----
        input['pred']: torch.tensor of torch.float [batch_size]
        input['target']: torch.tensor of torch.float [batch_size]
        output: torch.float
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input_pred = input['pred']
        self.input_target = input['target']
        self.output = output
        self.criterion = nn.MSELoss(reduction=reduction)
        self.normalize = normalize
        sizes[self.output] = []
    
    def forward(self, batch: dict, mode: str):
        input = batch[self.input_pred]
        target = batch[self.input_target]
        loss = self.criterion(input=input, target=target)
        if self.normalize:
            loss = loss / torch.var(target)
        batch[self.output] = loss
        return batch
    
class MultiMSECriterion(nn.Module):
    def __init__(self, logger: logging.Logger, sizes: dict, 
        input: dict = {'pred':None, 'target':None},
        output: str='MultiMSELoss', reduction: str='mean', normalize=False, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input_pred = input['pred']
        self.input_target = input['target']
        self.output = output
        self.criterion = nn.MSELoss(reduction='none')
        if reduction == 'mean':
            self.reduction = partial(torch.mean, dim=0)
        elif reduction == 'sum':
            self.reduction = partial(torch.sum, dim=0)
        elif reduction == 'none':
            self.reduction = EMPTY
        else:
            raise ValueError(f"Unsupported type of reduction: {reduction}")
        self.normalize = normalize
        sizes[self.output] = sizes[self.input_pred][1:]
    def forward(self, batch: dict, mode:str):
        input = batch[self.input_pred]
        target = batch[self.input_target]
        loss = self.reduction(self.criterion(input=input, target=target))
        if self.normalize:
            loss = loss / torch.var(target, dim=0)
        batch[self.output] = loss
        return batch

class MinusD_KLCriterion(nn.Module):
    def __init__(self, logger, sizes, input={'mu': 'mu', 'var': 'var'},
            output='-d_kl', **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input_mu = input['mu']
        self.input_var = input['var']
        self.output = output
        sizes[self.output] = []
    def forward(self, batch: dict, mode: str):
        mu = batch[self.input_mu]
        var = batch[self.input_var]
        batch[self.output] = 0.5*(torch.sum(mu**2)+torch.sum(var)-torch.sum(torch.log(var))-var.numel())
        return batch
# 今は使っていない。
import torch.nn as nn
from ..models import sigmoid 
from ..utils import check_leftargs
# LossManager
class LossManager(nn.Module):
    def __init__(self, logger, sizes, input, modes, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input = input
        self.modes = modes
    def forward(self, batch, mode):
        if mode in self.modes:
            batch['loss'] = batch['loss'] + batch[self.input]*self.factor(batch)
    def factor(self, batch):
        raise NotImplementedError
class ConstantLossManager(LossManager):
    name = 'lossmanager_constant'
    def __init__(self, logger, sizes, input, factor=1.0, modes=['train'], **kwargs):
        super().__init__(logger, sizes, input, modes, **kwargs)
        self.factor_ = factor
    def factor(self, batch):
        return self.factor_
class LinearLossManager(LossManager):
    name = 'lossmanager_linear'
    def __init__(self, logger, sizes, input, base_factor, low_step, low_factor, 
            annealing_step, high_step, high_factor, **kwargs):
        super().__init__(logger, sizes, input, modes=['train'], **kwargs)
        self.base_factor = float(base_factor)
        self.low_step = float(low_step)
        self.low_factor = low_factor
        self.annealing_step = float(annealing_step)
        self.high_factor = high_factor
        self.period = float(self.low_step + self.annealing_step + high_step)
    def factor(self, batch):
        step = batch['step'] % self.period
        if step <= self.low_step:
            return self.low_factor*self.base_factor
        elif step <= self.low_step+self.annealing_step:
            return (self.low_factor + (self.high_factor-self.low_factor) * \
                (step - self.low_step) / self.annealing_step)*self.base_factor
        else:
            return self.high_factor*self.base_factor

class SigmoidLossManager(LossManager):
    name = 'lossmanager_sigmoid'
    def __init__(self, logger, sizes, input, annealing_step, base_factor=1.0, low_step=0, low_factor=0.0,
            high_factor=1.0, high_step=0, sigmoid_range=5.0, **kwargs):
        """
        config:
          *base_factor: float(default=1.0)
          *low_step: int(default=0)
          *low_factor: float(default=0.0)
          annealing_step: int
          *high_factor: float(default=1.0)
          high_step: int(default=0)
          sigmoid_range: float(default=5.0)
        """
        super().__init__(logger, sizes, input, modes=['train'], **kwargs)
        self.base_factor = float(base_factor)
        self.low_step = float(low_step)
        self.low_factor = float(low_factor)
        self.annealing_step = float(annealing_step)
        self.high_factor = float(high_factor)
        self.period = float(low_step + annealing_step + high_step)
        self.sigmoid_range = float(sigmoid_range)
        self.sigmoid_coef = (self.high_factor - self.low_factor)/(sigmoid(self.sigmoid_range) - sigmoid(-self.sigmoid_range))
        self.sigmoid_bias = self.low_factor - sigmoid(-self.sigmoid_range)*self.sigmoid_coef
    def factor(self, batch):
        step =batch['step'] % self.period
        if step < self.low_step:
            factor = self.low_factor
        elif step < self.low_step + self.annealing_step:
            factor = sigmoid(((step-self.low_step)/self.annealing_step*2-1)*self.sigmoid_range)*self.sigmoid_coef+self.sigmoid_bias
        else:
            factor = self.high_factor
        return self.base_factor*factor
    
class LossNormalizer(nn.Module):
    name = 'loss_normalizer'
    def __init__(self, logger, sizes, input='loss', **kwargs):
        check_leftargs(self, logger, kwargs)
        self.input = input
    def forward(self, batch):
        input = batch[self.input]
        batch[self.input] = input / input.detach() * 10000
import torch
import torch.nn as nn

class GradCalculator:
    def __init__(self, loss_name, normalize=False, normalize_item=None):
        self.loss_name = loss_name
        self.normalize = normalize
        self.normalize_item = normalize_item
    
    def __call__(self, batch):
        if isinstance(self.loss_name, list):
            loss = sum(batch[n] for n in self.loss_name)
        else:
            loss = batch[self.loss_name]
        batch['loss'] = loss
        if self.normalize:
            loss = loss / loss.detach()
        if self.normalize_item is not None:
            loss = loss / batch[self.normalize_item]
        loss.backward()

# loss一定条件下でのみロスを計算する
class IntervalGradCalculator(GradCalculator):
    def __init__(self, target, min=-torch.inf, max=torch.inf, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.min = min
        self.max = max
    def __call__(self, batch):
        if self.min <= batch[self.target] <= self.max:
            super()(batch)
        
grad_calculator_type2class = {
    None: GradCalculator,
    'interval': IntervalGradCalculator
}
def get_grad_calculator(type=None, **kwargs) -> GradCalculator:
    return grad_calculator_type2class[type](**kwargs)

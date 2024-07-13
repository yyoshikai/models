import torch
import torch.nn as nn
from .. import register_module

@register_module
class GANCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, real_score: torch.Tensor, fake_score: torch.Tensor):
        real_target = torch.ones_like(real_score, device=real_score.device)
        fake_target = torch.zeros_like(fake_score, device=fake_score.device)
        score = torch.cat([real_score.ravel(), fake_score.ravel()])
        target = torch.cat([real_target.ravel(), fake_target.ravel()])
        return self.criterion(score, target)



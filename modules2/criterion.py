import torch
import torch.nn as nn
import torch.nn.functional as F
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


# 240718 graph autoencoderのため作成
@register_module
class EdgeCriterion(nn.Module):
    def __init__(self, node_pad_token, reduction='sum'):
        super().__init__()
        self.node_pad_token = node_pad_token
        self.reduction = reduction
    
    def forward(self, input, target, nodes):
        """
        input: [B, L, L, D]
        target: [B, L, L]
        nodes: [B, L]
        
        """
        B, L, _, D = input.shape
        pad_mask = nodes != self.node_pad_token
        pad_mask = torch.logical_and(pad_mask.unsqueeze(2), pad_mask.unsqueeze(1))
        return F.cross_entropy(input[pad_mask], target[pad_mask].to(torch.long), 
            reduction=self.reduction)

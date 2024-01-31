import torch
import torch.nn as nn
import torch.nn.functional as F

def off_diagonal(x: torch.Tensor):
    size = x.shape[0]
    return x.flatten()[:-1].view(size-1, size+1)[:, 1:].flatten()

class BarlowTwinsCriterion(nn.Module):
    def __init__(self, offdiag_weight):
        super().__init__()
        self.offdiag_weight = offdiag_weight
    
    def forward(self, input0: torch.Tensor, input1: torch.Tensor):
        """
        Parameters
        ----------
        input0: (torch.float)[batch_size, feature_size]
        input1: (torch.float)[batch_size, feature_size]
        
        Returns
        -------
        loss: float
        """
        batch_size, feature_size = input0.shape

        input0 = (input0 - input0.mean(dim=0)) / input0.std(dim=0)
        input1 = (input1 - input1.mean(dim=0)) / input1.std(dim=0)

        corr = torch.mm(input0.T, input1) / batch_size
        loss_on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
        loss_off_diag = off_diagonal(corr).pow_(2).sum()
        loss = loss_on_diag + loss_off_diag*self.offdiag_weight
        return loss

        

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

class MolCLIPCriterion(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, input0: torch.Tensor, input1: torch.Tensor):
        """
        Parameters
        ----------
        input0: (torch.float)[batch_size, feature_size]
        input1: (torch.float)[batch_size, feature_size]
        """
        sim = 0.5*(torch.sum(input0**2, dim=1).unsqueeze(1)
            +torch.sum(input1**2, dim=1)).unsqueeze(0) / self.temperature
            # [0のi batch, 1のi batch]
        diff = torch.mm(input0, input1.T) / self.temperature
            # [0のi batch, 1のi batch]

        loss0 = -F.softmax(sim, dim=1)*F.log_softmax(sim, dim=1)
        loss1 = -F.softmax(sim, dim=0)*F.log_softmax(sim, dim=0)
        return torch.sum(loss0+loss1)


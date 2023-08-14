import torch
import torch.nn as nn

class VAE(nn.Module):
    name = 'vae'
    def __init__(self, var_coef=1.0, eval_vae=False):
        """
        var_coef: float
        eval_vae: bool        
        """
        super().__init__()
        self.var_coef = var_coef
        self.eval_vae = eval_vae
        
    def forward(self, mode='train', mu=None, var=None, latent_size=None, device=None):
        """
        Parameters
        ----------
        mode: Either 'train', 'eval' or 'generate'
        """
        if mode == 'generate':
            return torch.randn(size=latent_size, device=device)
        else:
            if mode == 'train' or self.eval_vae:
                latent = mu + torch.randn(*mu.shape, device=mu.device)*torch.sqrt(var)*self.var_coef
            else:
                latent = mu
            return latent

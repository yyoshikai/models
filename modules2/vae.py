import torch
import torch.nn as nn
from models import register_module

@register_module
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

        # Add _device_param
        self._device_param = nn.Parameter(torch.zeros((0,)))
        def hook(model, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
            if prefix+'_device_param' not in state_dict:
                state_dict[prefix+'_device_param'] = model._device_param
        self._register_load_state_dict_pre_hook(hook, with_module=True)

    @property
    def device(self):
        return self._device_param.device
        
    def forward(self, mode='train', mu=None, var=None, latent_size=None, batch_size=None):
        """
        Parameters
        ----------
        mode: Either 'train', 'eval' or 'generate'
        """
        if mode == 'generate':
            return torch.randn(size=(batch_size, latent_size), device=self.device)
        else:
            if mode == 'train' or self.eval_vae:
                latent = mu + torch.randn(*mu.shape, device=mu.device)*torch.sqrt(var)*self.var_coef
            else:
                latent = mu
            return latent

# VAEと同じだが, eval()時はreparameterizationしないようにする。また複数サンプリング, 必ず反対側をサンプリングすることを可能にする。
@register_module
class VAE2(nn.Module):
    def __init__(self):
        
        super().__init__()
        self._device_param = nn.Parameter(torch.zeros((0,)))
        def hook(model, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
            if prefix+'_device_param' not in state_dict:
                state_dict[prefix+'_device_param'] = model._device_param
        self._register_load_state_dict_pre_hook(hook, with_module=True)

    @property
    def device(self):
        return self._device_param.device
        
    def forward(self, mode='train', 
            mu: torch.Tensor=None, var: torch.Tensor=None,
                train_augmentation=1, train_sample_opposite=False,
            latent_size=None, batch_size=None):
        """
        Parameters
        ----------
        mode: Either 'train', 'eval' or 'generate'
        """
        if mode == 'generate':
            return torch.randn(size=(batch_size, latent_size), device=self.device)
        else:
            if self.training:
                batch_size, latent_size = mu.shape
                latent = mu.repeat_interleave(train_augmentation, dim=0)
                if train_sample_opposite:
                    assert train_augmentation % 2 == 0
                    rand = torch.randn((int(batch_size*train_augmentation/2), latent_size), device=mu.device)
                    rand = torch.stack([rand, -rand], dim=1).reshape(-1, latent_size)
                else:
                    rand = torch.randn((batch_size*train_augmentation, latent_size), device=mu.device)
                latent = latent + rand*torch.sqrt(var).repeat_interleave(train_augmentation, dim=0)
            else:
                latent = mu
            return latent


@register_module
class MinusD_KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, var):
        return 0.5*(torch.sum(mu**2)+torch.sum(var)-torch.sum(torch.log(var))-var.numel())
    
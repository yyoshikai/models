import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from PIL import ImageOps


from .. import register_module

# 240424作成
@register_module
class ResNet18Backbone(nn.Sequential):
    def __init__(self, weights=None):
        if weights == 'imagenet1k_v1':
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        elif weights is not None:
            raise ValueError
        backbone = torchvision.models.resnet18(weights=weights)
        super().__init__(*list(backbone.children())[:-1])
    def forward(self, input: torch.Tensor):
        input = super().forward(input)
        return input.squeeze(-1).squeeze(-1)


class RandomSolarization(object):
    """Implementation of random image Solarization.

    Utilizes the integrated image operation `solarize` from Pillow. Solarization
    inverts all pixel values above a threshold (default: 128).

    Attributes:
        probability:
            Probability to apply the transformation
        threshold:
            Threshold for solarization.
    """

    def __init__(self, p: float = 0.5, threshold: int = 128):
        self.prob = p
        self.threshold = threshold

    def __call__(self, sample):
        """Solarizes the given input image

        Args:
            sample:
                PIL image to which solarize will be applied.

        Returns:
            Solarized image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # return solarized image
            return ImageOps.solarize(sample, threshold=self.threshold)
        # return original image
        return sample

transform_type2class = {
    cls.__name__: cls for cls in 
    [T.ToTensor, 
     T.CenterCrop, 
     T.Normalize, 

     # augmentation
     T.RandomHorizontalFlip,
     T.RandomGrayscale,
     T.ColorJitter,
     T.GaussianBlur,
     T.RandomRotation,
     T.RandomResizedCrop,
     RandomSolarization]
}
def get_transform(type, **kwargs):
    if type == 'RandomApply':
        return T.RandomApply(
            transforms = [get_transform(**t) for t in kwargs['transforms']],
            p=kwargs['p']
        )
    else:
        return transform_type2class[type](**kwargs)

@register_module
class Transform(nn.Module): # Wrap by nn.Module
    def __init__(self, transforms: list):
        super().__init__()
        self.transform = T.Compose([
            get_transform(**t) for t in transforms
        ])
    def forward(self, input):
        return self.transform(input)

def off_diag(tensor: torch.Tensor):
    size = tensor.shape[0]
    tensor = tensor.ravel()[:-1].reshape(size-1, size+1)[:, 1:].reshape(size, size-1)
    return tensor

@register_module
class TCCLRLoss(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau
    
    def forward(self, latent: torch.Tensor):
        """
        latent: [B*3, D]
        
        """
        bsz, _ = latent.shape
        assert bsz % 3 == 0
        bsz = int(bsz/3)

        latent = latent / torch.norm(latent, dim=-1, keepdim=True) # [B*3, D]
        sim = torch.mm(latent, latent.T) / self.tau # [B*3, B*4]
        sim_masked = off_diag(sim)
        softmax = torch.log(torch.sum(torch.exp(sim_masked), axis=1))

        numer = torch.sum(torch.diag(sim[:bsz, bsz:bsz*2])) + torch.sum(torch.diag(sim[bsz:bsz*2, :bsz]))
        numer += torch.sum(off_diag(sim[bsz*2:, bsz*2:])) / (bsz-1)
        loss = torch.sum(softmax) - numer
        loss /= bsz*3
        return loss
    

@register_module
class TCCLRLoss2(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau
    
    def forward(self, latent: torch.Tensor):
        """
        latent: [B*3, D]
        
        """
        bsz, _ = latent.shape
        assert bsz % 3 == 0
        bsz = int(bsz/3)

        latent = latent / torch.norm(latent, dim=-1, keepdim=True) # [B*3, D]
        sim = torch.mm(latent, latent.T) / self.tau # [B*3, B*4]
        sim_masked = off_diag(sim)
        softmax = torch.log(torch.sum(torch.exp(sim_masked), axis=1))

        numer = torch.sum(torch.diag(sim[:bsz, bsz:bsz*2])) + torch.sum(torch.diag(sim[bsz:bsz*2, :bsz]))
        numer += torch.sum(off_diag(sim[bsz*2:, bsz*2:])) / (bsz-1)
        loss = torch.sum(softmax) - numer / self.tau
        loss /= bsz*3
        return loss

class TripletLoss(nn.Module):
    def __init__(self, alpha: float, eps):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
    def forward(self, xp, xn):
        """
        Parameters
        ----------
        xp: [B, N, D]
        xn: [B, N, D] 

        B: batch size
        N: number of sampled images
        D: dimension of features 
        """
        bsz, n_sample, _ = xp.shape
        p_dist = 2 - 2*torch.bmm(xp, xp.mT) # [B, Np, Np]
        pn_dist = 2 - 2*torch.bmm(xp, xn.mT) # [B, Np, Np]
        ppn_dist = (self.alpha + p_dist.unsqueeze(3) - pn_dist.unsqueeze(2)) # [B, Np, Np, Np]
        ppn_mask = ppn_dist <= 0 # [B, Np, Np, Np]
        ppn_dist.masked_fill_(ppn_mask, 0)
        loss = torch.sum(ppn_dist, dim=-1) / (n_sample - torch.sum(ppn_mask, dim=-1)+self.eps)
        loss = torch.mean(loss)
        return loss

@register_module
class TCTripletLoss(nn.Module):
    def __init__(self, n_patch_per_wsi, alpha, eps=1e-9):
        super().__init__()
        self.n_patch_per_wsi = n_patch_per_wsi
        self.criterion = TripletLoss(alpha, eps)

    def forward(self, latent: torch.Tensor):
        bsz, _ = latent.shape
        bsz = int(bsz / (2*self.n_patch_per_wsi))
        latent = latent / torch.norm(latent, dim=-1, keepdim=True) # [B*2*Np, D]
        latent = latent.reshape(bsz, 2, self.n_patch_per_wsi, -1) # [B, 2, Np, D]
        t_latent, c_latent = latent[:,0], latent[:,1] # [B, Np, D]

        loss = self.criterion(t_latent, c_latent) + self.criterion(c_latent, t_latent)
        return loss

# Automated acquisition of explainable knowledge from unannotated histopathology images
# の再現
"""
以下適当にしたメモ
・padding_mode='reflect'か?
・初期値はtfだとweight=glorot uniform, bias=zerosがデフォ
・activationはなしとした。
    tfのデフォルトではなし。
・dropout=0.1とした。

"""
@register_module
class ImageAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input0 = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
        )
        self.input1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
        )
        self.input2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(45, 32, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 4, stride=4)
        )
        self.medium = nn.Sequential(
            nn.Linear(2048, 2048), 
            nn.Dropout(0.1)
        )
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=4), 
            # ここだけscale_factorより大きい方が良いと思った。
            nn.Conv2d(32, 32, 7, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=4), 
            nn.Conv2d(32, 32, 7, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 45, 7, padding='same', padding_mode='reflect'),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(45, 6, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(24, 3, 3, padding='same', padding_mode='reflect'),
        )


    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor[B, 3, 128, 128] 
        """
        B = x.shape[0]
        x = torch.cat([x, self.input0(x), self.input1(x), self.input2(x)], dim=1) # [B, 45, 128, 128]
        x = self.encoder(x) # [B, 32, 8, 8]
        z = x.reshape(B, -1) # [B, 2048]
        x = self.medium(z) # [B, 2048]
        x = x.reshape(B, 32, 8, 8)
        x = self.upsampler(x)
        x = self.decoder(x)
        
        return x, z

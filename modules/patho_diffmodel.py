"""
240217作成
patho/experiments/diffmodelをこちらに移動

"""
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision

def append_dropout(model, dropout):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, dropout)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=dropout))
            setattr(model, name, new)

class TCResNet(nn.Module):
    def __init__(self, ppi_c, ppi_t, backbone_path=None, dropout=0.0):
        super().__init__()
        self.ppi_c = ppi_c
        self.ppi_t = ppi_t
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        if dropout > 0.0:
            append_dropout(backbone, dropout)
        self.backbone_t = nn.Sequential(*list(backbone.children())[:-1])
        
        # load backbone
        if backbone_path is not None:
            state_dict_o = torch.load(backbone_path)
            state_dict = {}
            for key, value in state_dict_o.items():
                if 'backbone.' in key:
                    state_dict[key[9:]] = value
            self.backbone_t.load_state_dict(state_dict)

        # copy backbone for control
        self.backbone_c = deepcopy(self.backbone_t)
    def forward(self, x_t, x_c):
        x_t = self.backbone_t(x_t) # [B*P, D]
        x_c = self.backbone_c(x_c) # [B*P, D]
        x_t = x_t.squeeze(2).squeeze(2)
        x_c = x_c.squeeze(2).squeeze(2)
        _, d_model = x_t.shape
        x_t = torch.mean(x_t.reshape(-1, self.ppi_t, d_model), dim=1) # [B, D]
        x_c = torch.mean(x_c.reshape(-1, self.ppi_c, d_model), dim=1) # [B, D]
        x = torch.cat([x_t, x_c], dim=-1) # [B, D*2]
        return x
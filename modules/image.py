import yaml
from addict import Dict
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
from ..utils import check_leftargs
from ..models import Tunnel

transform_type2class = {
    'Normalize': T.Normalize,
    'RandomHorizontalFlip': T.RandomHorizontalFlip,
    'ColorJitter': T.ColorJitter,
    'GaussianBlur': T.GaussianBlur,
    'RandomResizedCrop': T.RandomResizedCrop,
    'ToTensor': T.ToTensor
}

def get_transform(tconfig):
    if isinstance(tconfig, str):
        return transform_type2class[tconfig]()
    else:
        type = tconfig.pop('type')
        if type == 'RandomApply':
            return T.RandomApply([
                get_transform(tconfig) for tconfig in tconfig['transforms'] 
            ], tconfig['p'])
        else:
            return transform_type2class[type](**tconfig)
def get_transforms(tconfigs):
    if isinstance(tconfigs, str):
        with open(f"transform_templates/{tconfigs}.yaml") as f:
            tconfigs = yaml.load(f, yaml.Loader)['transforms']
    return T.Compose(*[get_transform(tconfig) for tconfig in tconfigs])

class Module(nn.Module):
    def __init__(self, logger, sizes, modes, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes:
            return batch
        return self._forward(batch, mode)
    def _forward(self, batch, mode):
        raise NotImplementedError

class TransformsModule(Module):
    name = 'transforms'
    def __init__(self, input=None, output=None, transforms=None, modes=None, **kwargs):
        """
        Specify either template XOR transforms
        logger: logging.Logger
        sizes: {var_name(str): var_size(list)}
        transforms: Union[List[Dict], str]
        """
        super().__init__(modes=modes, **kwargs)
        self.input = input
        self.output = output
        self.transforms = get_transforms(transforms)

    def forward_(self, batch, mode):
        batch[self.output] = self.transforms(batch[self.input])
        return batch

backbone_type2class = {
    'resnet18': torchvision.models.resnet18
}
class FeatureExtractor(Module):
    name = 'feature_extractor'
    def __init__(self, logger, sizes, input, output, backbone, feature,
        modes=None, **kwargs):
        super().__init__(logger, sizes, modes, **kwargs)
        self.input = input
        self.output = output
        self.feature = feature
        backbone = backbone.copy()
        backbone = backbone_type2class[backbone.pop('type')](**backbone)
        self.extractor = create_feature_extractor(backbone, [feature])
    def _forward(self, batch, mode):
        batch[self.output] = self.extractor(batch[self.input])
        return batch

class BarlowTwinsHead(Module):
    name = 'barlowtwins_head'
    def __init__(self, logger, sizes, input, output, projection, modes=None, 
        **kwargs):
        super().__init__(logger, sizes, modes, **kwargs)
        self.input = input
        self.output = output
        self.projection = Tunnel(logger, projection, input_size=sizes[self.input])
        





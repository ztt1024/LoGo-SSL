import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class Backbone(nn.Module):
    """
    build the backbone
    """
    def __init__(self, arch='resnet34', fea_dim=128):
        """
        arch: arch of backbone (default: resnet34)
        fea_dim: output dimension of the projection head (default: 128)
        """
        super(Backbone, self).__init__()

        #create the backbone
        self.f = models.__dict__[arch]()

        #create the projection head
        self.prev_dim = self.f.fc.weight.shape[1]
        self.fea_dim = fea_dim
        self.f.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(self.prev_dim, self.prev_dim),
                                nn.ReLU(),
                                nn.Linear(self.prev_dim, self.fea_dim))


class Backbone_Cifar10(Backbone):
    """
    build the backbone in cifar10-style
    """
    def __init__(self, arch='resnet18', fea_dim=512):
        """
        arch: arch of backbone (default: resnet18)
        fea_dim: output dimension of the projection head (default: 512)
        """
        super(Backbone_Cifar10, self).__init__(arch, fea_dim)

        # create the backbone
        self.f = models.__dict__[arch]()

        # rebuild the backbone
        f_ls = []
        for name, module in self.f.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                f_ls.append((name, module))
        f_ls.append(('Flatten',nn.modules.Flatten(start_dim=1)))
        f_ls = OrderedDict(f_ls)
        self.f = nn.Sequential(f_ls)


class Encoder_Base(nn.Module):
    """
    build the encoder
    """
    def __init__(self, arch='resnet34', dim=128, dataset='imagenet100'):
        """
        dim: feature dimension (default: 128)
        dataset: dataset to pretrain on
        """
        # create the encoders
        super(Encoder_Base, self).__init__()
        if dataset == 'cifar10':
            model_backbone = Backbone_Cifar10(arch=arch, fea_dim=dim)
        else:
            model_backbone = Backbone(arch=arch, fea_dim=dim)

        self.f = model_backbone.f
        self.fc = model_backbone.fc


    def forward(self, x):
        feature = self.f(x)
        z = self.fc(feature)
        return feature, z

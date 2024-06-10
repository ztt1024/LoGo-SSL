import torch
import kornia as kn
from kornia.augmentation import Normalize, ColorJitter, RandomGrayscale, RandomHorizontalFlip

class aug_kornia(torch.nn.Module):
    def __init__(self):
        super(aug_kornia, self).__init__()
        self.nor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.colorjitter = ColorJitter(0.4, 0.4, 0.4, 0.1, p=.8)
        self.gray = RandomGrayscale(p=.2)
        self.hor = RandomHorizontalFlip()
        #self.gaussianblur = RandomGaussianBlur((23,23), (0.1, 2.0), p=1.0)

    def forward(self, img):
        img = self.colorjitter(img)
        img = self.gray(img)
        #img = self.gaussianblur(img)
        img = self.hor(img)
        img = self.nor(img)
        return img

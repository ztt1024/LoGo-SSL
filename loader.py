import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from PIL import ImageFilter, Image
import random

from torchvision.datasets import CocoCaptions
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS
from torchvision.datasets.vision import VisionDataset

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class coco(CocoCaptions):
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        target = ['1']

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CIFAR10_MC(CIFAR10):
    '''multicrop transform for CIFAR10'''

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        ls_pos = []
        if self.transform.__len__()==1:
            pos_1 = self.transform[0](img)
            pos_2 = self.transform[0](img)
            ls_pos = [pos_1, pos_2]
        elif self.transform.__len__()==2:
            ls_pos.append(self.transform[0](img))
            ls_pos.append(self.transform[0](img))
            ls_pos.append(self.transform[1](img))
            ls_pos.append(self.transform[1](img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ls_pos, target


class MultiCropsTransform():
    """Take multi-crops of one image as the query and key."""

    def __init__(self, transform):
        self.base_transform = transform

    def __call__(self, x):
        ls_pos = []
        if self.base_transform.__len__() == 1:
            pos_1 = self.base_transform[0](x)
            pos_2 = self.base_transform[0](x)
            ls_pos = [pos_1, pos_2]
        elif self.base_transform.__len__()==2:
            ls_pos.append(self.base_transform[0](x))
            ls_pos.append(self.base_transform[0](x))
            ls_pos.append(self.base_transform[1](x))
            ls_pos.append(self.base_transform[1](x))
        return ls_pos

class DatasetFolder_1k_100(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            in100_list = None
    ) -> None:
        super(DatasetFolder_1k_100, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.in100_list = in100_list
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = self.in100_list
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ReturnIndexImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform = None):
        super(ReturnIndexImageFolder, self).__init__(root=root,transform=transform)
        self.targets = [s[-1] for s in self.samples]
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexImageFolder, self).__getitem__(idx)
        return img, idx


class ReturnIndexImageFolder_1k(DatasetFolder_1k_100):
    def __init__(
            self,
            root: str,
            transform = None,
            in_100_list = None):
        super(ReturnIndexImageFolder_1k, self).__init__(root=root,transform=transform,in100_list=in_100_list)
        self.targets = [s[-1] for s in self.samples]
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexImageFolder_1k, self).__getitem__(idx)
        return img, idx


class ReturnIndexCIFAR10(CIFAR10):
    def __getitem__(self, index: int):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, index


def load_datasets(dataset='imagenet100',
                  datadir='/data',
                  args=None):
    if dataset == 'imagenet100' or dataset == 'imagenet1k_100' or dataset == 'coco':
        if dataset == 'coco':
            normalize = transforms.Normalize(mean=[0.471, 0.448, 0.408],
                                             std=[0.234, 0.239, 0.242])
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if args.mc:
            transform_g = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(args.global_scale[0], args.global_scale[1])),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])
            transform_l = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(args.local_scale[0], args.local_scale[1])),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])
            transform = [transform_g, transform_l]
        else:
            transform = [transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize
            ])
            ]
        traindir = os.path.join(datadir, 'train')
        #train_dataset = ImageNet_MC(traindir, transform=transform)
        if dataset == 'imagenet1k_100':
            f = open('imagenet100.txt')
            in100_list = []
            for classes in f.readlines():
                in100_list.append(classes[:-1])
            train_dataset = DatasetFolder_1k_100(traindir, transform=MultiCropsTransform(transform), in100_list=in100_list, loader=default_loader)
        elif dataset == 'imagenet100':
            train_dataset = ImageFolder(traindir, transform=MultiCropsTransform(transform))
        else:
            traindir = os.path.join(datadir, 'train2017')
            train_dataset = coco(root=traindir, annFile=os.path.join(datadir,'annotations/captions_train2017.json'),
                                 transform=MultiCropsTransform(transform))

    elif dataset == 'cifar10':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
        if args.mc:
            transform_g = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.global_scale[0], args.global_scale[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
            ])
            transform_l = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.local_scale[0], args.local_scale[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
            ])
            transform = [transform_g, transform_l]
        else:
            transform = [transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
            ])]
        train_dataset = CIFAR10_MC(root=datadir, train=True,  transform=transform)

    return train_dataset


def load_eval_datasets(dataset='imagenet100',
                  datadir='/data'):
    if dataset == 'imagenet100' or dataset == 'imagenet1k_100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.471, 0.448, 0.408],
        #                                  std=[0.234, 0.239, 0.242])
        transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
        traindir = os.path.join(datadir, 'train')
        valdir = os.path.join(datadir, 'val')
        if dataset == 'imagenet100':
            val_dataset = ReturnIndexImageFolder(valdir, transform=transform)
            train_dataset = ReturnIndexImageFolder(traindir, transform=transform)
        else:
            f = open('imagenet100.txt')
            in100_list = []
            for classes in f.readlines():
                in100_list.append(classes[:-1])
            val_dataset = ReturnIndexImageFolder_1k(valdir, transform=transform, in_100_list=in100_list)
            train_dataset = ReturnIndexImageFolder_1k(traindir, transform=transform, in_100_list=in100_list)

    elif dataset == 'cifar10':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = ReturnIndexCIFAR10(root=datadir, train=True,  transform=transform)
        val_dataset = ReturnIndexCIFAR10(root=datadir, train=False, transform=transform)

    return train_dataset, val_dataset


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
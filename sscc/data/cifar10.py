import torch
import torchvision
import os
import numpy as np
import pdb
import logging
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from torchvision.transforms.transforms import ToPILImage, ToTensor

from sscc.data.randaugment import RandAugmentCIFAR10

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

from torchvision import transforms
from torch.utils import data
from sscc.data.images import ImageDataset

class CIFAR10(ImageDataset):
    """
    """
    base_folder = 'cifar10'

    def __init__(self,
                 root,
                 part,
                 val_size,
                 num_constraints,
                 k,
                 seed=1337,
                 transform=None,
                 download=True,
                 fold=0,
                 **kwargs):
        super(CIFAR10, self).__init__(root, part=part,
                                      val_size=val_size,
                                      num_constraints=num_constraints,
                                      k=k,
                                      seed=seed,
                                      transform=transform,
                                      download=download,
                                      fold=fold)

        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = os.path.join(self.root, self.base_folder)
        if download:
            self.download()
        self.x, self.y, self.c = self.load_dataset(part=self.part, fold=self.fold)

    def __len__(self):
        """as the __getitem__() must work differently for train and val/test,
        the __len__() must be specified such that the indeces are only sampled from the desired sample space.
        For train: __len__() corresponds to constraints df (C_train.csv)
        For val/test: __len() corresponds to the total num of obs available.
        """
        if self.part == 'train':
            return len(self.c)
        else:
            return self.x.shape[0]

    def download(self):

        if not self.should_download():
            if self.part == 'train':
                _, y, _ = self.load_dataset(part=self.part, fold=self.fold)
                c_df_train = self.build_constraints(y, self.num_constraints, seed=self.seed+self.fold)
                c_df_train.to_csv(f"{self.dataset_path}/fold_{self.fold}/C_train.csv")
                return
            return

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        trainset = torchvision.datasets.CIFAR10(root=self.dataset_path,
                                                train=True,
                                                download=True,
                                                transform=transform)
        testset = torchvision.datasets.CIFAR10(root=self.dataset_path,
                                               train=False,
                                               download=True,
                                               transform=transform)

        # norm to [0, 1]
        X_train = trainset.data.swapaxes(2, 3).swapaxes(1, 2) / 255.
        Y_train = np.array(trainset.targets)
        X_test = testset.data.swapaxes(2, 3).swapaxes(1, 2) / 255.
        Y_test = np.array(testset.targets)

        self._split_and_save(x_test=X_test,
                             x_train=X_train,
                             y_test=Y_test,
                             y_train=Y_train)

        os.system('rm {}/cifar-10-python.tar.gz; rm -rf {}/cifar-10-batches-py'.format(self.dataset_path, self.dataset_path))


normalize_cifar10 = normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

transforms_cifar10_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_cifar10,
])

transforms_cifar10_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize_cifar10,
])

transforms_cifar10_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_cifar10,
])

transforms_cifar10_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.125,0.125)),
    transforms.ToTensor(),
    normalize_cifar10
])


transforms_cifar10_strong = transforms.Compose([
                                    transforms.ToPILImage(),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.RandomCrop(size=32,
                                    #                     padding=int(32*0.125),
                                    #                     padding_mode='reflect'),
                                    RandAugmentCIFAR10(n=2, m=10),
                                    transforms.ToTensor(),
                                    normalize_cifar10,
                                    ])



# SIMPLE AND EFFICIENT AUGMENTS - DO NOT CHANGE SPEED EITHER
# transforms_cifar10_weak = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     normalize_cifar10,
# ])

# transforms_cifar10_strong = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     normalize_cifar10,
# ])
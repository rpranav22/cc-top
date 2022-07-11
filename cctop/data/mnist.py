import torch
import torchvision
import os

from torchvision import transforms
from cctop.data.images import ImageDataset
from cctop.data.randaugment import RandAugmentFMNIST

class MNIST(ImageDataset):
    """
    k: controls amount of neighbors per sampled constrain aka the connectedness
        k = 1: super narrow, no point part of > 1 pair
        k = 1000: super dense, 1000 constraints connected to the 1 same data point
        default: None aka random sampling
    """
    base_folder = 'mnist'

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
        super(MNIST, self).__init__(root, part=part,
                                    val_size=val_size,
                                    num_constraints=num_constraints,
                                    k=k,
                                    seed=seed,
                                    transform=transform,
                                    fold=fold,
                                    download=download)

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


        trainset = torchvision.datasets.MNIST(root=self.dataset_path,
                                             train=True,
                                             download=True,
                                             transform=None)
        testset = torchvision.datasets.MNIST(root=self.dataset_path,
                                            train=False,
                                            download=True,
                                            transform=None)

        X_train = trainset.data.unsqueeze(1) / 255.
        Y_train = trainset.targets
        X_test = testset.data.unsqueeze(1) / 255.
        Y_test = testset.targets

        self._split_and_save(X_test, X_train, Y_test, Y_train)

        os.system(f'rm -rf {self.dataset_path}/MNIST/')

normalize_mnist = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

transforms_mnist_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_mnist
])

transforms_mnist_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize_mnist
])

transforms_mnist_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_mnist
])

transforms_mnist_strong = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    RandAugmentFMNIST(n=5, m=10),
    transforms.ToTensor(),
    normalize_mnist
])
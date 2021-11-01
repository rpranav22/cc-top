import os
import numpy as np
import pandas as pd
import torch
import cv2
import random
import pdb
import requests

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils import data
from sklearn.model_selection import train_test_split
from sscc.data.images import ImageDataset
from sscc.data.randaugment import RandAugmentYALEB

class YaleBExt(ImageDataset):
    """
    """
    base_folder = 'yalebext'
    url = 'http://vision.ucsd.edu/extyaleb/ExtendedYaleB.tar.bz2'
    filename = "ExtendedYale.zip"
    id_list = ['B11', 'B12', 'B13', 'B15', 'B16', 'B17',
               'B18','B19', 'B20', 'B21', 'B22', 'B23',
               'B24', 'B25', 'B26', 'B27', 'B28', 'B29',
               'B30', 'B31', 'B32', 'B33', 'B34', 'B35',
               'B36', 'B37', 'B38', 'B39']
    crop_anchors = {'B11': (60, 180),
                    'B12': (60, 180),
                    'B13': (40, 180),
                    'B15': (50, 180),
                    'B16': (60, 180),
                    'B17': (60, 180),
                    'B18': (70, 180),
                    'B19': (60, 180),
                    'B20': (60, 180),
                    'B21': (60, 220),
                    'B22': (60, 180),
                    'B23': (60, 180),
                    'B24': (40, 160),
                    'B25': (40, 180),
                    'B26': (70, 180),
                    'B27': (60, 170),
                    'B28': (100, 180),
                    'B29': (40, 200),
                    'B30': (60, 180),
                    'B31': (90, 160),
                    'B32': (60, 180),
                    'B33': (60, 180),
                    'B34': (40, 180),
                    'B35': (60, 180),
                    'B36': (60, 180),
                    'B37': (60, 200),
                    'B38': (40, 160),
                    'B39': (40, 170)
                    }
    crop_length = 340

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
        super(YaleBExt, self).__init__(root, part=part,
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

        np.random.seed(self.seed)

        for id in self.id_list:
            url = f"http://vision.ucsd.edu/extyaleb/ExtendedYaleBZip/yale{id}.zip"
            filename = f"yale{id}.zip"
            download_and_extract_archive(url, self.root, filename=filename)


        files = {}
        counter = 0
        for id in self.id_list:
            extracted_path = os.path.join(self.root, f'yale{id}')
            for (dirpath, dirnames, filenames) in os.walk(extracted_path):
                filedirs = []
                for name in filenames:
                    filedirs.append(os.path.join(dirpath, name))
                files[f"{counter}"] = filedirs
            counter += 1

        # download cascade classifier from OpenCV
        URL = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalcatface.xml"
        response = requests.get(URL)
        with open('haarcascade_frontalface_default.xml', 'wb') as file:
            file.write(response.content)

        labels = []
        images = []
        for key, id in zip(files.keys(), self.crop_anchors.keys()):
            for f in files[key]:
                try:
                    img = cv2.imread(f, -1)
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(float) / 255.0
                    if img.shape != (1, 480, 640):
                        continue
                    y1, x1 = self.crop_anchors[id][0], self.crop_anchors[id][1]
                    img = img[:, y1:(y1+self.crop_length), x1:(x1+self.crop_length)]
                    img = np.transpose(img, axes=(1, 2, 0))
                    img = cv2.resize(img, (120, 120))
                    img = np.expand_dims(img, axis=0)

                    images.append(img)
                    labels.append(int(key))
                except:
                    continue

        images = np.stack(images)
        labels = np.array(labels)
        # split in train and test
        X_train, X_test, Y_train, Y_test = train_test_split(images, labels,
                                                            test_size=self.val_size,
                                                            random_state=self.seed,
                                                            stratify=labels)
        self._split_and_save(x_test=X_test,
                             x_train=X_train,
                             y_test=Y_test,
                             y_train=Y_train)

        # remove files
        for id in self.id_list:
            os.system(f"rm -rf {self.root}/yale{id}; rm {self.root}/yale{id}.zip")


transforms_yaleb_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transforms_yaleb_strong = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    RandAugmentYALEB(n=5, m=10),
    transforms.ToTensor()
])

transforms_yaleb_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
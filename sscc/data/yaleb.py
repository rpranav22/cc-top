import os
import numpy as np
import pandas as pd
import torch
import cv2
import random

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import save_image
from torch.utils import data
from sklearn.model_selection import train_test_split
from sscc.data.images import ImageDataset

class YaleB(ImageDataset):
    """
    """
    base_folder = 'yaleb'
    url = "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
    filename = "CroppedYale.zip"

    def __init__(self,
                 root,
                 part,
                 val_size,
                 num_constraints,
                 k,
                 seed=1337,
                 transform=None,
                 download=True,
                 **kwargs):
        super(YaleB, self).__init__(root, part=part,
                                    val_size=val_size,
                                    num_constraints=num_constraints,
                                    k=k,
                                    seed=seed,
                                    transform=transform,
                                    download=download)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = os.path.join(self.root, self.base_folder)
        if download:
            self.download()

        self.x, self.y, self.c = self.load_dataset(part=self.part)

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
                _, y, _ = self.load_dataset(part=self.part)
                c_df_train = self.build_constraints(y, self.num_constraints)
                c_df_train.to_csv(f"{self.dataset_path}/C_train.csv")
                return
            return

        np.random.seed(self.seed)
        download_and_extract_archive(self.url, self.root, filename=self.filename)
        extracted_path = os.path.join(self.root, 'CroppedYale')

        # store filedirs in dict
        files = {}
        counter = 0
        for (dirpath, dirnames, filenames) in os.walk(extracted_path):
            filedirs = []
            for name in filenames:
                filedirs.append(os.path.join(dirpath, name))
            files[f"{counter}"] = filedirs
            counter += 1

        # load data and store in in array
        labels = []
        images = []

        for key in files.keys():
            for f in files[key]:
                try:
                    img = cv2.imread(f, -1)
                    img = np.expand_dims(img, axis=0)
                    img = torch.tensor(img).float() / 255.0
                    if img.shape != (1, 192, 168):
                        continue
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
        os.system(f'rm {self.root}/CroppedYale.zip; rm -rf {self.root}/CroppedYale')
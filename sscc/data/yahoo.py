# logger = logging.getLogger(__name__)

import os
from sscc.data.texts import TextDataset
from kaggle.api.kaggle_api_extended import KaggleApi
from datasets import load_dataset_builder, get_dataset_config_names, load_dataset
import pandas as pd

class yahoo(TextDataset):
    base_folder = 'yahoo'

    def __init__(self, 
                root: str, part: str, val_size: float, 
                num_constraints: int, k: int, seed: int = 1337, test_size: float=0.2,
                clean_text: bool = True, remove_stopwords: bool = True, 
                download: bool = True, **kwargs):
        super(yahoo, self).__init__(root, part, 
                                        val_size, num_constraints, k, 
                                        seed=seed, test_size=test_size,
                                        clean_text=clean_text, remove_stopwords=remove_stopwords,
                                        download=download, **kwargs)
        # self.fold = fold
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ['KAGGLE_USERNAME'] = "pranavragupathy"
        os.environ['KAGGLE_KEY'] = "8f45e8901dd909e24068d603e68157bd"

        self.api = KaggleApi()
        self.api.authenticate()

        self.dataset_path = os.path.join(self.root, self.base_folder)
        
        if download:
            self.download()
        self.x, self.y, self.c = self.load_dataset(part=self.part, clean_text=self.clean_text, remove_stopwords=self.remove_stopwords)
    
  
    def __getitem__(self, index):
        return super().__getitem__(index)

    def download(self):

        if self.should_download():
            print('downloading from kaggle')
            self.api.dataset_download_files('jarupula/yahoo-answers-dataset', path=self.dataset_path, unzip=True)

        self.train_data = pd.read_csv(f"{self.dataset_path}/train.csv", header=None)
        self.test_data = pd.read_csv(f"{self.dataset_path}/test.csv", header=None)

        X_train = list(self.train_data[1].astype(str).values)
        y_train = list(self.train_data[0].values)
        y_train = [t - 1 for t in y_train]

        X_test = list(self.test_data[1].astype(str).values)
        y_test = list(self.test_data[0].values)
        y_test = [t - 1 for t in y_test]
        
        print('\n'*2, len(X_train), len(y_train))

        self._split_and_save(X_train, y_train, y_test, X_test)
        


        


# logger = logging.getLogger(__name__)

import os
from cctop.data.texts import TextDataset
from datasets import load_dataset

class dbpedia(TextDataset):
    base_folder = 'dbpedia'

    def __init__(self, 
                root: str, part: str, val_size: float, 
                num_constraints: int, k: int, seed: int = 1337, test_size: float=0.2,
                clean_text: bool = True, remove_stopwords: bool = True, 
                download: bool = True, **kwargs):
        super(dbpedia, self).__init__(root, part, 
                                        val_size, num_constraints, k, 
                                        seed=seed, test_size=test_size,
                                        clean_text=clean_text, remove_stopwords=remove_stopwords,
                                        download=download, **kwargs)
        # self.fold = fold
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = os.path.join(self.root, self.base_folder)
        
        if download:
            self.download()
        self.x, self.y, self.c = self.load_dataset(part=self.part, clean_text=self.clean_text, remove_stopwords=self.remove_stopwords)
    
  
    def __getitem__(self, index):
        return super().__getitem__(index)

    def download(self):

        if not self.should_download():
            # if self.part == 'train':
            #     _, y, _= self.load_dataset(part=self.part)
            #     # c_df_train = self.build_constraints(y, self.num_constraints, seed=self.seed)
            #     # c_df_train.to_csv(f"{self.dataset_path}/C_train.csv")
            #     return
            return

        self.data = load_dataset("dbpedia_14")

        X_train = self.data['train']['content']
        y_train = self.data['train']['label']

        X_test = self.data['test']['content']
        y_test = self.data['test']['label']
        
        print('\n'*2, len(X_train), len(y_train))

        self._split_and_save(X_train, y_train, y_test, X_test)
        


        


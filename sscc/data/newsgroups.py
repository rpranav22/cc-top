# logger = logging.getLogger(__name__)

import os
from sscc.data.texts import TextDataset
from datasets import load_dataset_builder, get_dataset_config_names, load_dataset

class newsgroups(TextDataset):
    base_folder = 'newsgroups'

    def __init__(self, 
                root: str, part: str, val_size: float, 
                num_constraints: int, k: int, seed: int = 1337, test_size: float=0.2,
                clean_text: bool = True, remove_stopwords: bool = True, is_tensor=True,
                download: bool = True, **kwargs):
        super(newsgroups, self).__init__(root, part, 
                                        val_size, num_constraints, k, 
                                        seed=seed, test_size=test_size,
                                        clean_text=clean_text, remove_stopwords=remove_stopwords,
                                        is_tensor=is_tensor, download=download, **kwargs)
        # self.fold = fold
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = os.path.join(self.root, self.base_folder)
        self.is_tensor = is_tensor
        if download:
            self.download()
        self.x, self.y = self.load_dataset(part=self.part, clean_text=self.clean_text, remove_stopwords=self.remove_stopwords, is_tensor=self.is_tensor)
    
    def __len__(self):
        """as the __getitem__() must work differently for train and val/test,
        the __len__() must be specified such that the indeces are only sampled from the desired sample space.
        For train: __len__() corresponds to constraints df (C_train.csv)
        For val/test: __len() corresponds to the total num of obs available.
        """
        # if self.part == 'train':
        #     return len(self.c)
        # else:
        return len(self.y)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def download(self):

        if not self.should_download():
            if self.part == 'train':
                _, y= self.load_dataset(part=self.part)
                # c_df_train = self.build_constraints(y, self.num_constraints, seed=self.seed)
                # c_df_train.to_csv(f"{self.dataset_path}/C_train.csv")
                return
            return

        newsgroup_configs = get_dataset_config_names("newsgroup")
        newsgroup_configs = [x for x in newsgroup_configs if x.startswith('19997')]

        self.metadata = dict.fromkeys(range(20))


        for i, d in enumerate(newsgroup_configs):
            categories = d[6:].split('.')
            config_name = d
            self.metadata[i] = {'categories': categories, 'config_name': config_name}

        # print("metadata: ", self.metadata)

        self.data = dict.fromkeys(range(20))

        for i in range(20):
            self.data[i] = load_dataset('newsgroup', self.metadata[i]['config_name'])

        # print(self.data)

        X_train = []
        y_train = []
        for k, v in self.data.items():
            for text in v['train']:
                # print(text['text'])
                X_train.append(text['text'])
                y_train.append(k)
        
        print('\n'*2, len(X_train), len(y_train))

        self._split_and_save(X_train, y_train)
        


        


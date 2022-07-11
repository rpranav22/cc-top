import os
from cctop.data.texts import TextDataset
from datasets import load_dataset

class newsgroups(TextDataset):
    base_folder = 'newsgroups'

    def __init__(self, 
                root: str, part: str, val_size: float, 
                num_constraints: int, k: int, seed: int = 1337, test_size: float=0.2,
                clean_text: bool = True, remove_stopwords: bool = True, 
                download: bool = True, **kwargs):
        super(newsgroups, self).__init__(root, part, 
                                        val_size, num_constraints, k, 
                                        seed=seed, test_size=test_size,
                                        clean_text=clean_text, remove_stopwords=remove_stopwords,
                                        download=download, **kwargs)
        self.dataset_path = os.path.join(self.root, self.base_folder)
        if download:
            self.download()
        self.x, self.y, self.c = self.load_dataset(part=self.part, clean_text=self.clean_text, remove_stopwords=self.remove_stopwords)
    
  
    def __getitem__(self, index):
        return super().__getitem__(index)

    def download(self):

        if not self.should_download():
            if self.part == 'train':
                _, y, _= self.load_dataset(part=self.part)
                return
            return

        newsgroup_configs = get_dataset_config_names("newsgroup")
        newsgroup_configs = [x for x in newsgroup_configs if x.startswith('19997')]

        self.metadata = dict.fromkeys(range(20))


        for i, d in enumerate(newsgroup_configs):
            categories = d[6:].split('.')
            config_name = d
            self.metadata[i] = {'categories': categories, 'config_name': config_name}

        self.data = dict.fromkeys(range(20))

        for i in range(20):
            self.data[i] = load_dataset('newsgroup', self.metadata[i]['config_name'])

        X_train = []
        y_train = []
        for k, v in self.data.items():
            for text in v['train']:
                X_train.append(text['text'])
                y_train.append(k)
        
        self._split_and_save(X_train, y_train)
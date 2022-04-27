from operator import itemgetter
import re
import shutil
from typing import List
import numpy as np
import pandas as pd
import torch
import os
import pdb
import pickle

from torch.utils import data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import BertTokenizerFast, BertForSequenceClassification

class TextDataset(data.Dataset):
    """Superclass for the text datasets

    Contains the `build_constraints()` method that samples constraints from originally labeled datasets
    """    
    def __init__(self,
                 root: str,
                 part: str,
                 val_size: float,
                 num_constraints: int,
                 k: int,
                 max_length: int,
                 constrained_clustering: bool=False,
                 seed: int=1337,
                 test_size: float=0.2,
                 clean_text: bool = True, 
                 remove_stopwords: bool = True,
                 download: bool=True,
                 **kwargs):
        """Text Data Base Class

        Args:
            root (str): the root data path
            part (str): train/val/test dataset 
            val_size (int): size of the validation data set
            num_constraints (int): amount of constraints to be sampled for the train dataset
            k (int): the amount of neighbors per constraint to be sampled. Controls the underlying constraint graph: NULL means random sampling, k=50 means that one randomly selected data point has pairwise connections to 50 other data points (equiv. to a very dense graph)
            seed (int, optional): Seed for constraint sampling to stay reproducible. Defaults to 1337.
            download (bool, optional): should data be downloaded. Defaults to True.
        """               
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.part = part
        self.dataset_path = os.path.join(self.root, self.base_folder)
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=True)
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length
        self.clean_text = clean_text
        self.constrained_clustering = constrained_clustering
        self.remove_stopwords = remove_stopwords
        self.k = k
        self.num_constraints = num_constraints
        self.seed = seed
        if 'topic_discovery' in kwargs:
            self.topic_discovery = kwargs['topic_discovery']
        if 'new_samples' in kwargs:
            self.new_samples = kwargs['new_samples']
            self.phase = 2
        if 'num_samples' in kwargs:
            self.num_samples = kwargs['num_samples']
            self.phase = 1
        else:
            self.num_samples = None
        if 'train_type' in kwargs:
            self.train_type = kwargs['train_type']
        if 'new_split' in kwargs:
            self.new_split = kwargs['new_split']
        if 'model_uri' in kwargs:
            self.model_uri = kwargs['model_uri']

    @property
    def size(self):
        if self.part == 'train' and self.constrained_clustering:
            return self.c.shape[0]
        else:
            return self.x.shape[0]

    @property
    def num_classes(self):
        return self.num_classes

    def __len__(self):

        if self.part == 'train' and self.constrained_clustering:
            return self.c.shape[0]
        else:
            return len(self.x)

    def __getitem__(self, index):
        """the iterator over indices work differently for train and val/test data.
        For train: we only want to iterate over the pre-specified constraints.
                   There one sample is one constraint consisting of two observations (as defined in C_train.csv)
                   The final constraint matrix is then built in data.uitls.constrained_collate_fn()
        For val/test: we want to iterate over the whole data.
                   For this, one sample corresponds to one observation.
                   The constraints are then built in data.utils.supervised_collate_fn()
        """
        if torch.is_tensor(index):
            index = index.tolist()
        # pdb.set_trace()

        if self.part == 'train' and self.constrained_clustering:
            # print(f'index: {index}\n {self.c.info()}')
            # pdb.set_trace()
            constraint_info = self.c.iloc[index, :]

            i, j = constraint_info['i'], constraint_info['j']
            c_ij = constraint_info['c_ij']
            y_i, y_j = constraint_info['y_i'], constraint_info['y_j']

            assert y_i == self.y[i]

            # pdb.set_trace()

            return self.x[i], self.x[j], y_i, y_j, c_ij

            
        else:
            return self.x[index], self.y[index]
            
        

    def load_dataset(self, part='train', clean_text=True, remove_stopwords=True):

        path = os.path.join(self.root, self.base_folder)

        with open(f"{path}/X_{part}", 'rb') as fp:
            x = pickle.load(fp)
        with open(f"{path}/Y_{part}", 'rb') as fp:
            y = pickle.load(fp)

        y = np.asarray(y).astype(int)

        assert len(x) == len(y)

        if clean_text:
            x = self.clean_texts(x)
    

        # to avoid building constraints for the val and test set because it takes very long
        part='train'
        constraints = pd.read_csv(f"{path}/C_{part}.csv")

        return x, y , constraints
    
    def tokenize_text(self, texts):
        return self.tokenizer.encode_plus(texts, truncation=True, padding=True, max_length=self.max_length)

    def divide_dataset_by_classes(self, x, y, excluded_classes=[10, 11, 12, 13], num_samples=None):
        
        print("\n\n\ntrying topic discovery initial size: {}\n\n\n".format(len(y)))
        # excluded_classes = list(range(10))
        
        print('\nexcluded classes: ', excluded_classes)

        labelled_set = []
        unlabelled_set = []

        for x_i, y_i in zip(x,y):
            if y_i in excluded_classes:
                unlabelled_set.append((x_i,y_i))
            else:
                labelled_set.append((x_i,y_i))

        
        x_l,y_l = zip(*labelled_set)
        x_ul,y_ul = zip(*unlabelled_set)

        if self.phase == 2:
            # pdb.set_trace()
            cdf_uri = "/".join(self.model_uri.split('/')[:-1]) + "/c_df_train.csv"
            c_df = pd.read_csv(cdf_uri)
            constrained_samples = np.unique(c_df[['i', 'j']].values)

            x_ul = [x_samp for i, x_samp in enumerate(x_ul) if not i in constrained_samples]
            y_ul = [y_samp for i, y_samp in enumerate(y_ul) if not i in constrained_samples]

            if self.new_split == '2v2':
                x_1, _, y_1, _ = train_test_split(x_l, y_l, train_size=2000, stratify=y_l)
                x_2, _, y_2, _ = train_test_split(x_ul, y_ul, train_size=2000, stratify=y_ul)

                # split_2 = StratifiedShuffleSplit(n_splits=1, train_size=2000, test_size=2000)
                # pdb.set_trace()
                # train_indices = split_2.split(x_l, y_l)
                # x_1, y_1 = x_l[train_indices], y_l[train_indices]

                # for _, test_index in split_2.split(x_ul, y_ul):
                #     x_2, y_2 = x_ul[test_index], y_ul[test_index]
                # pdb.set_trace()
                if self.train_type == 'finetune':
                    x_s = list(itemgetter(*constrained_samples)(x))
                    y_s = list(itemgetter(*constrained_samples)(y))
                    x = x_s + x_1 + x_2
                    y = y_s + y_1 + y_2
               



        if self.phase == 1:
            x = x_l
            y = y_l

        # pdb.set_trace()

        print(f'final train size: {len(y)}')

        
        return x, y
    
    def _split_and_save(self, x_train, y_train, y_test=None, x_test=None):
        """
        helper method to split train/test data into train/val/test data and store them as .npy arrays
        samples constraints from the labeled datasets on the fly

        Input: 
            * labeled train dataset (X, Y)
            * labeled test dataset (X, Y)

        Output:
            * labeled train dataset (X, Y, C)
            * labeled val dataset (X, Y, C)
            * labeled test dataset (X, Y, C)
        
        Where C refers to the constraint-matrix
        """

        # folds = 5
        # for fold in range(folds):
        #     # reset seed
        np.random.seed(self.seed)

        
        # os.mkdir(dataset_path)

        if self.topic_discovery:
            if self.phase == 1:
                x_train, y_train = self.divide_dataset_by_classes(x_train, y_train)
                # x_val, y_val = self.divide_dataset_by_classes(x_val, y_val)
                if x_test:
                    x_test, y_test = self.divide_dataset_by_classes(x_test, y_test)



        if not y_test and not x_test:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=self.test_size,
                                                            random_state=self.seed,
                                                            train_size=self.num_samples,
                                                            stratify=y_train)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                            test_size=self.val_size,
                                                            random_state=self.seed,
                                                            train_size=self.num_samples,
                                                            stratify=y_train)
        if self.topic_discovery:
            if self.phase == 2:

                x_train, y_train = self.divide_dataset_by_classes(x_train, y_train, list(range(10)), 4000)

        # build constraints
        c_df_train = self.build_constraints(np.array(y_train).astype(np.int32), int(self.num_constraints), seed=self.seed)
        # c_df_val = self.build_constraints(np.array(y_val).astype(np.int32), int(self.num_constraints), seed=self.seed)
        # c_df_test = self.build_constraints(np.array(y_test).astype(np.int32), int(self.num_constraints), seed=self.seed)

        # store sampled constraints
        c_df_train.to_csv(f"{self.dataset_path}/C_train.csv")
        # c_df_val.to_csv(f"{self.dataset_path}/C_val.csv")
        # c_df_test.to_csv(f"{self.dataset_path}/C_test.csv")

        


        if not self.constrained_clustering:
            constrained_samples = np.unique(c_df_train[['i', 'j']].values)
            print(f'\n\n Total no. of samples used for the constraints is {len(constrained_samples)}')
            print(f"\n length of xtrain was {len(x_train)}\nif we want to sample {self.num_constraints} constraints, we would be using {len(constrained_samples)} samples from the data for the baselines.\n\n")
            x_train = list(itemgetter(*constrained_samples)(x_train))
            y_train = list(itemgetter(*constrained_samples)(y_train))
            print(f'length of x_train is {len(x_train)}\n\n')

        # store split data as pickle file
        with open(f"{self.dataset_path}/X_train", 'wb') as fp:
            pickle.dump(x_train, fp)
        with open(f"{self.dataset_path}/X_val", 'wb') as fp:
            pickle.dump(x_val, fp)
        with open(f"{self.dataset_path}/X_test", 'wb') as fp:
            pickle.dump(x_test, fp)
        with open(f"{self.dataset_path}/Y_train", 'wb') as fp:
            pickle.dump(y_train, fp)
        with open(f"{self.dataset_path}/Y_val", 'wb') as fp:
            pickle.dump(y_val, fp)
        with open(f"{self.dataset_path}/Y_test", 'wb') as fp:
            pickle.dump(y_test, fp)
            

    def should_download(self) -> bool:
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            return True
        else:
            if not any(os.scandir(self.dataset_path)):
                print('path exists but empty af')
                return True
            else:
                if self.part=='train':
                    shutil.rmtree(self.dataset_path)
                    print(f'deleteing the contents of this path {self.dataset_path} and rebuilding dataset with constraints.')
                    os.makedirs(self.dataset_path)
                    return True
                else:
                    print(f"checking for {self.part} here. If exists, not building again")
                    return False

    def clean_texts(self, data):
        """
        Removes emails and newline characters
        """
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        return data

    def build_constraints(self, y: np.ndarray, num_constraints: int, seed: int=0) -> np.ndarray:
        """Samples random pairwise constraints.

        self.k controls the amount of neigbors per selected sample to control connectedness of the graph
            - k = NULL: randomly sampled pairs, semi-connected graph
            - k = 1: very dense, no connected parts
            - k = 1000: very connected, 1000 constraints connected to 1 single data sample

        Args:
            y (np.ndarray): vector of class labels
            num_constraints (int): amount of constraints to be sampled
            seed (int, optional): reproc. seed. Defaults to 0.

        Returns:
            (np.ndarray): the constraint matrix C
        """        
        if self.k:
            assert len(y) * self.k >= num_constraints, f"too few obs: {len(y)} for required num_constraints {num_constraints} given parameter k: {self.k}"

        np.random.seed(seed)
        print(f'=== SEED {seed}, FOLD ')
        idx_sample_basis = np.arange(0, len(y))

        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []

        while num_constraints > 0:
            if self.k:
                # select k partners per sample
                tmp1 = np.random.choice(a=idx_sample_basis, size=1)[0]
                # make sure no idx1 is sampled twice
                idx_sample_basis = idx_sample_basis[idx_sample_basis != tmp1]

                k_cnt = 0
                # now select k partners for that sample tmp1
                # high k -> many partners, low k -> few to 1 partner
                while k_cnt < self.k:
                    tmp2 = np.random.choice(a=idx_sample_basis, size=1)[0]
                    if tmp1 == tmp2:
                        # not a valid choice => a constraint with itself is meaningless
                        continue
                    if y[tmp1] == y[tmp2]:
                        ml_ind1.append(tmp1)
                        ml_ind2.append(tmp2)
                    else:
                        cl_ind1.append(tmp1)
                        cl_ind2.append(tmp2)

                    num_constraints -= 1
                    k_cnt += 1
            else:
                # randomly select two samples from the labeled dataset
                tmp1 = np.random.randint(0, len(y) - 1)
                tmp2 = np.random.randint(0, len(y) - 1)
                if tmp1 == tmp2:
                    # not a valid choice => a constraints with itself is meaningless
                    continue
                if y[tmp1] == y[tmp2]:
                    # Must link constraint
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    # Cannot link constraint
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)

                num_constraints -= 1

        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
        # apply transitivity closure of ML and entailment of CL
        # fills the underlying constraint graph and makes sure we use all information that we have
        # ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, len(y))

        total_constraints = ml_ind1.shape[0] + cl_ind1.shape[0]

        constraints_i = np.hstack((ml_ind1, cl_ind1)).astype(np.int32)
        constraints_j = np.hstack((ml_ind2, cl_ind2)).astype(np.int32)
        
        c_df = pd.DataFrame(index=np.arange(total_constraints),
                            columns=['idx', 'part', 'i', 'j', 'y_i', 'y_j', 'c_ij'])

        c_df['idx'] = np.arange(total_constraints)
        c_df['part'] = self.part
        c_df['i'] = constraints_i
        c_df['j'] = constraints_j
        c_df['y_i'] = y[constraints_i]
        c_df['y_j'] = y[constraints_j]
        c_df['c_ij'] = np.where(c_df['y_i'] == c_df['y_j'], 1, -1)

        print(f'\nFrom {self.part}, I sampled {self.num_constraints} constraints with k={self.k} \nresulting in {len(c_df)} constraints after TC/CE calculation\n')

        return c_df
    
    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number

        # Return
            transitive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)

        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

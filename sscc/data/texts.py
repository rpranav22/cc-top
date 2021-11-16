import re
from typing import List
import numpy as np
import pandas as pd
import torch
import os
import pdb
import pickle

from torch.utils import data
from sklearn.model_selection import train_test_split
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
                 seed: int=1337,
                 max_length: int=512,
                 test_size: float=0.2,
                 clean_text: bool = True, 
                 remove_stopwords: bool = True,
                 is_tensor: bool=True,
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
        # self.transform = transform
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=True)
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length
        self.clean_text = clean_text
        self.remove_stopwords = remove_stopwords
        self.k = k
        self.num_constraints = num_constraints
        self.seed = seed

    @property
    def size(self):
        # if self.part == 'train':
        return self.c.shape[0]
        # else:
        #     return self.x.shape[0]

    @property
    def num_classes(self):
        return len(self.y)

    def __len__(self):
        # if self.part == 'train':
        return len(self.x)
        # else:
        #     return self.x.shape[0]

    def __getitem__(self, index):
        """the iterator over indices work differently for train and val/test data.
        For train: we only want to iterate over the pre-specified constraints.
                   There one sample is one constraint consisting of two observations (as defined in C_train.csv)
                   The final constraint matrix is then built in data.uitls.constrained_collate_fn()
        For val/test: we want to iterate over the whole data.
                   For this, one sample corresponds to one observation.
                   The constraints are then built in data.utils.supervised_collate_fn()
        """
        item = {k: torch.tensor(v[index]) for k, v in self.x.items()}
        item["labels"] = torch.tensor([self.y[index]])
        return item
        

    def load_dataset(self, part='train', clean_text=True, remove_stopwords=True, is_tensor=True):

        path = os.path.join(self.root, self.base_folder)

        with open(f"{path}/X_{part}", 'rb') as fp:
            x = pickle.load(fp)
        with open(f"{path}/Y_{part}", 'rb') as fp:
            y = pickle.load(fp)

        y = np.asarray(y).astype(int)

        assert len(x) == len(y)

        if clean_text:
            x = self.clean_texts(x)
        # constraints = pd.read_csv(f"{path}/C_{part}.csv")
        if is_tensor:
            x = self.tokenize_text(x)

        return x, y #, constraints
    
    def tokenize_text(self, texts):
        return self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)

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
        #     np.random.seed(self.seed)

        dataset_path = os.path.join(self.root, self.base_folder)
        # os.mkdir(dataset_path)

        if not y_test and not x_test:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=self.test_size,
                                                            random_state=self.seed,
                                                            stratify=y_train)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                            test_size=self.val_size,
                                                            random_state=self.seed,
                                                            stratify=y_train)
        
        # # build constraints
        # c_df_train = self.build_constraints(np.array(y_train), self.num_constraints, seed=self.seed)
        # c_df_val = self.build_constraints(np.array(y_val), self.num_constraints, seed=self.seed)
        # c_df_test = self.build_constraints(np.array(y_test), self.num_constraints, seed=self.seed)

        # # store sampled constraints
        # c_df_train.to_csv(f"{dataset_path}/C_train.csv")
        # c_df_val.to_csv(f"{dataset_path}/C_val.csv")
        # c_df_test.to_csv(f"{dataset_path}/C_test.csv")

        # store split data as pickle file
        with open(f"{dataset_path}/X_train", 'wb') as fp:
            pickle.dump(x_train, fp)
        with open(f"{dataset_path}/X_val", 'wb') as fp:
            pickle.dump(x_val, fp)
        with open(f"{dataset_path}/X_test", 'wb') as fp:
            pickle.dump(x_test, fp)
        with open(f"{dataset_path}/Y_train", 'wb') as fp:
            pickle.dump(y_train, fp)
        with open(f"{dataset_path}/Y_val", 'wb') as fp:
            pickle.dump(y_val, fp)
        with open(f"{dataset_path}/Y_test", 'wb') as fp:
            pickle.dump(y_test, fp)
            

    def should_download(self) -> bool:
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            return True
        else:
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
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, len(y))

        total_constraints = ml_ind1.shape[0] + cl_ind1.shape[0]

        constraints_i = np.hstack((ml_ind1, cl_ind1))
        constraints_j = np.hstack((ml_ind2, cl_ind2))

        c_df = pd.DataFrame(index=np.arange(total_constraints),
                            columns=['idx', 'part', 'i', 'j', 'y_i', 'y_j', 'c_ij'])

        c_df['idx'] = np.arange(total_constraints)
        c_df['part'] = self.part
        c_df['i'] = constraints_i
        c_df['j'] = constraints_j
        c_df['y_i'] = y[constraints_i]
        c_df['y_j'] = y[constraints_j]
        c_df['c_ij'] = np.where(c_df['y_i'] == c_df['y_j'], 1, -1)

        print(f'\nI sampled {self.num_constraints} constraints with k={self.k} \nresulting in {len(c_df)} constraints after TC/CE calculation\n')

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

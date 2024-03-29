import pdb
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from torch.utils.data.dataloader import default_collate
from transformers.file_utils import is_torch_available
from transformers import BertTokenizerFast
from cctop.data.agnews import agnews
from cctop.data.banking import banking
from cctop.data.cifar10 import CIFAR10, transforms_cifar10_train, transforms_cifar10_test
from cctop.data.cifar20 import CIFAR20
from cctop.data.mnist import MNIST, transforms_mnist_train, transforms_mnist_test
from cctop.data.fashionmnist import FASHIONMNIST, transforms_fmnist_train, transforms_fmnist_test
from cctop.data.trec import trec
from cctop.data.tweeteval import tweeteval
from cctop.data.yaleb import YaleB
from cctop.data.yalebextend import YaleBExt, transforms_yaleb_train
from cctop.data.newsgroups import newsgroups
from cctop.data.dbpedia import dbpedia

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

def constrained_collate_fn(batch, params):
    """
    Collates the constrained samples only
    one sample of a batch consists of 5 elements: 
        1) xi
        2) xj
        3) yi
        4) yj
        5) cij
    """
    # https://stackoverflow.com/questions/62669261/how-to-encode-multiple-sentences-using-transformers-berttokenizer

    # from timeit import default_timer as timer; start = timer()
    transposed_data = list(zip(*batch))
    data = [default_collate(b) for b in transposed_data]
    # print(f"\n\n\n\nprinting from inside the constrained collate fn \n\n\n {len(data), type(data)}, and params {params}\n\n")
    
    x_i = list(data[0])
    x_j = list(data[1])
    y_i = data[2].type(torch.int32)
    y_j = data[3].type(torch.int32)
    c_ij = data[4]

    # notes = list(zip(x_i, x_j))
    targets = torch.cat((y_i, y_j), dim=0)
    # target_j = torch.tensor(self.y[j])
    # print( y_i, y_j, c_ij, targets.shape)
    # target = target_i.transpose()

    encoding_xi = tokenizer.batch_encode_plus(
    x_i,
    add_special_tokens=True,
    max_length=params['max_length'],
    return_token_type_ids=True,
    truncation=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    )    

    encoding_xj = tokenizer.batch_encode_plus(
    x_j,
    add_special_tokens=True,
    max_length=params['max_length'],
    return_token_type_ids=True,
    truncation=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    )   
            
    # rearrange pre-specified constraints to make them trainable!
    train_target, eval_target = prepare_task_target(targets, c_ij)
    
    stacked_input = torch.cat((encoding_xi['input_ids'], encoding_xj['input_ids']), dim=0)
    stacked_attention = torch.cat((encoding_xi['attention_mask'], encoding_xj['attention_mask']), dim=0)
    stacked_tokens = torch.cat((encoding_xi['token_type_ids'], encoding_xj['token_type_ids']), dim=0)
    # print(f"\n\n\n\nprinting from inside the collate fn \n\n\n {targets.shape}, {encoding_xi['input_ids'].shape}, {stacked_input.shape }\n\n\n")

    return {
        #'text': note,
        'label': torch.tensor(targets, dtype=torch.long),
        'input_ids': (stacked_input),
        'attention_mask': (stacked_attention),
        'token_type_ids': (stacked_tokens),
        'train_target': train_target
    }

def supervised_collate_fn(batch, params):
    """
    Collates supervised, labeled samples
    """
    transposed_data = list(zip(*batch))
    data = [default_collate(b) for b in transposed_data]
    # print(f"\n\n\n\nprinting from inside the collate fn \n\n\n {len(data), type(data[0])}, and params {params}\n\n")

    notes = list(data[0])
    notes = [str(x) for x in notes]
    targets = data[1]

    encoding = tokenizer.batch_encode_plus(
    notes,
    add_special_tokens=True,
    max_length=params['max_length'],
    return_token_type_ids=True,
    truncation=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    )    
    # print(f"\n\n\n\nprinting from inside the collate fn \n\n\n {targets}, {encoding['input_ids'].shape }\n\n\n")
    train_target, eval_target = prepare_supervised_task_target(target=data[1])
    # pdb.set_trace()
    return {
        #'text': note,
        'label': torch.tensor(targets, dtype=torch.long),
        'input_ids': (encoding['input_ids']),
        'attention_mask': (encoding['attention_mask']),
        'token_type_ids': (encoding['token_type_ids']),
        'train_target': (train_target)
    }

#TODO: come up with a collate logic for ConstraintMatch
def constraint_match_collate_fn(batch, data_collate=default_collate):
    """
    Collates the unconstrained samples
    """
    transposed_data = list(zip(*batch))
    data = [data_collate(b) for b in transposed_data]

    return {'weakly_aug': data[0].double(), 'strongly_aug': data[1].double(), 'y': data[2]}


def prepare_supervised_task_target(target):
    """
    """
    train_target = Class2Simi(x=target)
    # train_target = target
    eval_target = target

    return train_target.detach(), eval_target.detach()

def prepare_task_target(target, constraints):
    """
    """
    train_target = Constraints2Simi(x=target, constraints=constraints)
    eval_target = target

    return train_target.detach(), eval_target.detach()

def Constraints2Simi(x, constraints, mask=None):
    """
    """
    n = int(x.shape[0]/2)
    #TODO: find out whether this should be 0 or 1?
    trivial_constraints = torch.eye(n) * 0
    prespecified_constraints = torch.diag(constraints)
    out = torch.vstack([torch.hstack([trivial_constraints, prespecified_constraints]), torch.hstack([prespecified_constraints, trivial_constraints])])
    out = out.double()

    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

def Class2Simi(x, mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1) == n, 'Dimension of Label is not correct'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    out = out.double() #hingeloss require double type

    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    # labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    # print(len(labels), len(preds), acc, " ___________ acc")
    return {
        'accuracy': acc,
    }

def get_data(root, params, log_params, part):
    """
    """

    if params['dataset'] == 'newsgroups':
        data = newsgroups(root=root,
                       part=part,
                       **params
                       )
    
    elif params['dataset'] == 'agnews':
        data = agnews(root=root,
                       part=part,
                       **params)

    elif params['dataset'] == 'trec':
        data = trec(root=root,
                       part=part,
                       **params)
    
    elif params['dataset'] == 'dbpedia':
        data = dbpedia(root=root,
                       part=part,
                       **params)
    
    elif params['dataset'] == 'tweeteval':
        data = tweeteval(root=root,
                       part=part,
                       **params)

    elif params['dataset'] == 'banking':
        data = banking(root=root,
                        part=part,
                        **params)

    return data
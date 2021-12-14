import torch
import torch.nn as nn
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F
from sscc.data.utils import PairEnum
import random

class SupervisedPLM(nn.Module):
    def __init__(self, model, loss) -> None:
        super(SupervisedPLM, self).__init__()
        self.max_length = 512
        self.model_name = "bert-base-uncased"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    
    # def forward(self, batch, **kwargs):
    #     """
    #     """
    #     # train_target = pairwise constraints;
    #     # len(train_target) = batch_size**2
    #     # eval_target = actual labeles used for evaluation
        
    #     #need to modify collate function for this to work 
    #     # images = batch['texts'].to(self.device)

    #     outputs = self.model(batch)

    #     # out = F.softmax(outputs, dim=1)
    #     return outputs

    def forward(self, *input, **kwargs):
            # print(type(input))
            return self.model(*input, **kwargs)
    

    
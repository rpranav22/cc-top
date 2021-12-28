import torch
import torch.nn as nn
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F
from sscc.data.utils import PairEnum
import random

class ConstrainedClustering(nn.Module):
    def __init__(self, model, loss) -> None:
        super(ConstrainedClustering, self).__init__()
        self.max_length = 512
        self.model_name = "bert-base-uncased"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.pre_classifier = nn.Linear(self.model.bert.config.hidden_size, self.model.bert.config.hidden_size)
        self.classifier = nn.Linear(self.model.bert.config.hidden_size, 20)
        self.dropout = nn.Dropout(self.model.bert.config.hidden_dropout_prob)
        self.relu =  nn.ReLU()


    def forward(self, *input, **kwargs):
            # print("This is in supervised model", type(input), type(kwargs), kwargs.keys())
            # print(f"this is in supervised model class forward and output is of type {type(input)} and shape {len(input)}")

            outputs = self.model(*input, **kwargs)
            
            hidden_state = outputs[0]  # (bs, seq_len, dim)
            # print(f"last hidden state classifier {hidden_state.shape}")
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = self.relu(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            print(f"before classifier {pooled_output.shape}")
            logits = self.classifier(pooled_output)  # (bs, dim)

            # print(f"same place but what type {type(logits)} and shape {logits.shape} are the logits here :{logits[0]}")

            logits_softmax = F.softmax(logits, dim=1)
            print(f"same place shape {logits_softmax.shape} and the tensor softie itself {logits_softmax[0]}")

            return logits_softmax
    

    
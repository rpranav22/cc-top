from torch import nn as nn
import numpy as np
# from transformers import BertForSequenceClassification
# import transformers
from transformers import BertModel, BertConfig, RobertaConfig, RobertaModel

class BertForClassification(nn.Module):
  
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1, **kwargs):
        super(BertForClassification, self).__init__()
        self.num_labels = kwargs['num_labels']
        self.config = BertConfig()
        self.bert = BertModel.from_pretrained(kwargs['base_model'], config=self.config)

        
        # nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, labels):

        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)

        
        return outputs
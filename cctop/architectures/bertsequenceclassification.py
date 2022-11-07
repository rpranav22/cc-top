from torch import nn as nn
from transformers import BertModel, BertConfig

class BertForClassification(nn.Module):
  
    def __init__(self, **kwargs):
        super(BertForClassification, self).__init__()
        self.num_labels = kwargs['num_classes']
        self.config = BertConfig()
        self.bert = BertModel.from_pretrained(kwargs['base_model'], config=self.config)

    def forward(self, input_ids, attention_mask, label):
        
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)
        
        return outputs
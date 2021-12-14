from torch import nn as nn
from transformers import BertForSequenceClassification
import transformers
from transformers import BertModel, BertConfig

class BertForClassification(nn.Module):
  
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1, **kwargs):
        super(BertForClassification, self).__init__()
        self.num_labels = kwargs['num_labels']
        self.config = BertConfig()
        self.bert = BertModel.from_pretrained(kwargs['base_model'], config=self.config)
        # self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.classifier = nn.Linear(hidden_size, kwargs['num_labels'])

        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.relu =  nn.ReLU()
        # nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits
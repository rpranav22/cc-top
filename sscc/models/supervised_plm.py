import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
import random

class SupervisedPLM():
    def __init__(self) -> None:
        self.max_length = 512
        self.model_name = "bert-base-uncased"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=20).to(self.device)

    
from bertopic import BERTopic
import os

class BERTopic(BERTopic):
    def __init__(self, embedding_model=None, num_classes: int = None):
        super(BERTopic, self).__init__()
        self.embedding_model = embedding_model
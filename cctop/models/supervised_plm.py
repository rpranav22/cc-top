from typing import Any
import torch
import torch.nn as nn
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
from cctop.metrics import Evaluator
from torch.nn import functional as F
import os
import pandas as pd
import pdb
import tempfile

class SupervisedPLM(nn.Module):
    def __init__(self, model, loss, num_classes, **kwargs) -> None:
        super(SupervisedPLM, self).__init__()
        # self.max_length = 512
        self.model_name = kwargs['arch']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model #.to(self.device)

        self.pre_classifier = nn.Linear(self.model.bert.config.hidden_size, self.model.bert.config.hidden_size)
        self.classifier = nn.Linear(self.model.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(self.model.bert.config.hidden_dropout_prob)
        self.relu =  nn.ReLU()


    def forward(self, *input, **kwargs):

            outputs = self.model(*input, **kwargs)
            
            hidden_state = outputs[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = self.relu(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            logits = self.classifier(pooled_output)  # (bs, dim)
            # pdb.set_trace()
            return logits
    
    def loss_function(self, out, batch, **kwargs):
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(out.view(-1, self.model.num_labels), batch['label'])

        return {'loss': loss}
    
    def evaluate(self, labels: Any, eval_dataloader: Any, confusion: Any=Evaluator(20), part: str='val',current_epoch: int=0, logger: Any=None, true_k: int=20):
        """Evaluate model on any dataloader during training and testings

        Args:
            eval_dataloader (Any): data loader for evaluation
            confusion (Any, optional): the evaluator object as definsed in cctop.metrics. Defaults to Evaluator(10).
            part (str, optional): train/test/val . Defaults to 'val'.
            logger (Any, optional): any pl logger. Defaults to None.
            true_k (int, optional): true clusters, important for confusion matrix plotting. Defaults to 10.

        Returns:
            None
        """   
        if part == 'test': y, pred = [], [] 
        for i, batch in enumerate(eval_dataloader):
            # input_ids = batch['input_ids']
            # label = batch['label']
            # attention_mask = batch['attention_mask']

            # with torch.no_grad():
            #     outs = self.forward(input_ids, attention_mask, label)

            confusion.add(batch, labels[i])
            if part == 'test':
                y.extend(labels[i].cpu().detach().numpy())
                pred.append(batch.cpu().detach().numpy())

        if part == 'test': 
            pred = np.concatenate(pred, axis=0)


        confusion.optimal_assignment(confusion.k)
        print('\n')
        print(f"{part} ==> clustering scores: {confusion.clusterscores()}")
        print(f"{part} ==> accuracy: {confusion.acc()}")
        # store in dictionary for mlflow logging
        eval_results = {f'{part}_acc': confusion.acc()}
        eval_results.update({f'{part}_{key}': value for key, value in confusion.clusterscores().items()})

        if part != 'train':
            epoch_string=f'{current_epoch}'.zfill(4)
            logger.experiment.log_figure(figure=confusion.plot_confmat(title=f'{part} set, epoch {current_epoch}',
                                                                       true_k=true_k),
                                         artifact_file=f'confmat_{part}_{epoch_string}.png',
                                         run_id=logger.run_id)
        if part == 'test':
            # log the test predictions
            # pdb.set_trace()
            # final_results = {f'yhat_p_{cl}': pred[:, cl] for cl in range(pred.shape[0])}
            final_results = {}
            final_results['y'] = y
            final_results['yhat'] = np.argmax(pred, 1)
            final_results = pd.DataFrame(final_results)
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, 'test_preds.csv')
                final_results.to_csv(storage_path)
                logger.experiment.log_artifact(local_path=storage_path, run_id=logger.run_id)

        return eval_results
    
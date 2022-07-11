import torch.nn as nn
import torch
import pdb
import numpy as np
import tempfile
import os 
import pandas as pd

from torch.nn import functional as F
from typing import Any
from cctop.losses import KCL, MCL
from cctop.metrics import Evaluator
from cctop.data.utils import PairEnum

from timeit import default_timer as timer


class Supervised(nn.Module):
    """
    """
    def __init__(self, model, loss):
        super(Supervised, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = KCL().to(self.device) if loss == 'KCL' else MCL().to(self.device)

    def forward(self, batch, **kwargs):
        """
        """
        # train_target = pairwise constraints;
        # len(train_target) = batch_size**2
        # eval_target = actual labeles used for evaluation
        images = batch['images'].to(self.device)
        logits = self.model(images)
        out = F.softmax(logits, dim=1)

        return {'out': out}

    def loss_function(self, outs, batch, **kwargs):
        """
        """
        out = outs['out']
        train_target = batch['train_target']
        prob1, prob2 = PairEnum(out)

        loss = self.criterion(prob1, prob2, simi=train_target)

        return {'loss': loss}

    def evaluate(self, eval_dataloader, confusion=Evaluator(10), part='val', logger=None):
        """
        """
        for i, batch in enumerate(eval_dataloader):
            outs = self.forward(batch=batch)
            confusion.add(outs['out'], batch['eval_target'])

        confusion.optimal_assignment(confusion.k)
        print(f"{part} ==> clustering scores: {confusion.clusterscores()}")
        print(f"{part} ==> accuracy: {confusion.acc()}")
        # store in dictionary for mlflow logging
        eval_results = {f'{part}_acc': confusion.acc()}
        eval_results.update({f'{part}_{key}': value for key, value in confusion.clusterscores().items()})

        if part != 'train':
            epoch_string=f'{self.epoch}'.zfill(4)
            logger.experiment.log_figure(figure=confusion.plot_confmat(title=f'{part} set, epoch {self.epoch}'),
                                         artifact_file=f'confmat_{part}_{epoch_string}.png',
                                         run_id=logger.run_id)
        return eval_results


    def evaluate(self, eval_dataloader: Any, confusion: Any=Evaluator(10), part: str='val', logger: Any=None, true_k: int=10):
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
            outs = self.forward(batch=batch)
            confusion.add(outs['out'], batch['eval_target'])
            if part == 'test':
                y.extend(batch['eval_target'].detach().cpu().numpy())
                pred.append(outs['out'].detach().cpu().numpy())

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
            epoch_string=f'{self.epoch}'.zfill(4)
            logger.experiment.log_figure(figure=confusion.plot_confmat(title=f'{part} set, epoch {self.epoch}',
                                                                       true_k=true_k),
                                         artifact_file=f'confmat_{part}_{epoch_string}.png',
                                         run_id=logger.run_id)
        if part == 'test':
            # log the test predictions
            final_results = {f'yhat_p_{cl}': pred[:, cl] for cl in range(pred.shape[1])}
            final_results['y'] = y
            final_results['yhat'] = np.argmax(pred, 1)
            final_results = pd.DataFrame(final_results)
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, 'test_preds.csv')
                final_results.to_csv(storage_path)
                logger.experiment.log_artifact(local_path=storage_path, run_id=logger.run_id)

        return eval_results


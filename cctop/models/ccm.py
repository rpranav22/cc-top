import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import tempfile
import os 
import pandas as pd

from typing import Any
from torch.nn import functional as F
from cctop.losses import KCL, MCL
from cctop.metrics import Evaluator
from cctop.data.utils import Class2Simi, PairEnum, prepare_supervised_task_target
from scipy.special import entr


class CCM(nn.Module):
    """Constrained ConstraintMatch Model 
    """
    def __init__(self, model, loss):
        super(CCM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = KCL().to(self.device) if loss == 'KCL' else MCL().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch, **kwargs):          
        if len(batch) == 3:
            images = batch['images'].to(self.device)
            logits = self.model(images)
            out = F.softmax(logits, dim=1)
            return {'out': out}
        else:
            # forward pass for constrained observations
            images = batch['supervised_train']['images'].to(self.device)
            logits = self.model(images)
            out_supervised = F.softmax(logits, dim=1)

            # forward pass for weakly augmented unconstrained data
            img_weak = batch['cm_train']['weakly_aug'].to(self.device)
            logits = self.model(img_weak)
            out_weak = F.softmax(logits, dim=1)

            # forward pass for strongly augmented unconstrained data
            img_strong = batch['cm_train']['strongly_aug'].to(self.device)
            logits = self.model(img_strong)
            # we use the softmax probs to create pseudo-constraints
            out_strong = F.softmax(logits, dim=1)

            return {'out': out_supervised, 'out_weak': out_weak, 'out_strong': out_strong}

    def loss_function(self, outs, batch, loss_type, lambda_unsup, threshold, **kwargs):
        """
        """
        if len(batch) == 3:
            # catch the constrained only case 
            out = outs['out']
            train_target = batch['train_target']
            prob1, prob2 = PairEnum(out)
            loss = self.criterion(prob1=prob1, prob2=prob2, simi=train_target)

            return {'loss': loss}
        else:
            out = outs['out']
            out_weak = outs['out_weak']
            out_strong = outs['out_strong']
            train_target = batch['supervised_train']['train_target']

           # 1) supervised loss
            prob1, prob2 = PairEnum(out)
            loss_sup = self.criterion(prob1=prob1, prob2=prob2, simi=train_target)

            ### CCM: create pseudo-constraints from pseudo-labels
            if loss_type == 'ccm_1':
                # i) select confident pseudo labels pl
                max_prob, max_class = torch.max(out_weak, dim=1)
                out_strong_confident = out_strong[max_prob > threshold]
                pl = max_class[max_prob > threshold]

                # ii) calculate pseudo constraints from them 
                # set some to 0.0 if we do not want to use them/ artificially balance 
                pc_target = Class2Simi(torch.tensor(pl))

                # iii) track pl stats
                k = out_weak.shape[1]
                confusion = Evaluator(k=k)
                confusion.add(output=out_weak, target=batch['cm_train']['y'])
                confusion.optimal_assignment(confusion.k)
                pl_acc = confusion.acc()
                if int(sum(max_prob > threshold)) > 1:
                    # cluster assignment for quality of CONFIDENT ONLY weak pseudo labels
                    confusion = Evaluator(k=k)
                    confusion.add(output=out_weak[max_prob > threshold], target=batch['cm_train']['y'][max_prob > threshold])
                    confusion.optimal_assignment(confusion.k)
                    pl_conf_acc = confusion.acc()
                else: 
                    pl_conf_acc = 0     

                # 2) unsupervised loss via pseudo-constraints 
                prob1_saug, prob2_saug = PairEnum(out_strong_confident)
                loss_unsup = self.criterion(prob1=prob1_saug, prob2=prob2_saug, simi=pc_target)
                if loss_unsup.isnan(): loss_unsup = torch.zeros_like(loss_unsup)
                loss = loss_sup + lambda_unsup * loss_unsup

                return {'loss': loss,
                        'loss_sup': loss_sup,
                        'loss_unsup': loss_unsup * lambda_unsup, 
                        'pl_ratio': sum(max_prob > threshold) / len(max_prob),
                        'pl_acc': torch.tensor(pl_acc),
                        'pl_conf_acc': torch.tensor(pl_conf_acc),
                        'pc_ml_ratio': sum(pc_target == 1.0) / len(pc_target) if len(pc_target) > 0.0 else torch.tensor(0.0),
                        }

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


    def pl_stats(self, cm_train_gen, threshold, logger, step):
        """
        """
        outs_weak = []
        outs_strong = []
        y = []
        for i, batch in enumerate(cm_train_gen):
            out_weak = F.softmax(self.model(batch['weakly_aug'].to(self.device)), dim=1)
            out_strong = F.softmax(self.model(batch['strongly_aug'].to(self.device)), dim=1)
            outs_weak.append(out_weak.detach().cpu().numpy())
            outs_strong.append(out_strong.detach().cpu().numpy())
            y.extend(batch['y'].detach().cpu().numpy())

        outs_weak = np.vstack(outs_weak)
        outs_strong = np.vstack(outs_strong)


        # 1.) average entropy
        entr_weak_mean = entr(outs_weak).sum(axis=1).mean()
        entr_strong_mean = entr(outs_strong).sum(axis=1).mean()
        logger.experiment.log_metric(key='weak_entropy',
                                     value=entr_weak_mean,
                                     step=step,
                                     run_id=logger.run_id)
        logger.experiment.log_metric(key='strong_entropy',
                                     value=entr_strong_mean,
                                     step=step,
                                     run_id=logger.run_id)

        # 2.) distribution of max softmax predictions (all obs)
        max_class_weak = np.argmax(outs_weak, axis=1)
        max_class_strong = np.argmax(outs_strong, axis=1)
        max_prob_weak = np.amax(outs_weak, axis=1)
        max_prob_strong = np.amax(outs_strong, axis=1)

        fig, axs = plt.subplots(2)
        fig.suptitle(f"distribution of max. softmax pred.: epoch {step}")
        axs[0].hist(max_prob_weak, bins=1000, density=True, histtype='stepfilled')
        axs[0].set_title("Weakly augmented samples")
        axs[1].hist(max_prob_strong, bins=1000, density=True, histtype='stepfilled')
        axs[1].set_title("Strongly augmented samples")
        fig.tight_layout()
        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'max_softmax_{step}.png',
                                     run_id=logger.run_id)

        # 3.) distribution of max softmax predictions (considered obs only => passed threshold
        counts_weak = np.unique(outs_weak.argmax(1), return_counts=True)
        counts_strong = np.unique(outs_strong.argmax(1), return_counts=True)
        y_all = np.unique(y, return_counts=True)

        fig, axs = plt.subplots(3)
        fig.suptitle(f"Predictive class distribution \nnon-annotated samples in epoch {step}")

        axs[0].bar(x=counts_weak[0], height=counts_weak[1])
        axs[0].set_xticks(counts_weak[0])
        axs[0].set_title("Weakly augmented samples")

        axs[1].bar(x=counts_strong[0], height=counts_strong[1])
        axs[1].set_xticks(counts_strong[0])
        axs[1].set_title("Strongly augmented samples")

        axs[2].bar(x=y_all[0], height=y_all[1])
        axs[2].set_xticks(y_all[0])
        axs[2].set_title("True labels")

        fig.tight_layout()

        # ax.set_xticks([np.arange(len(out_strong))])
        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'pred_dist_{step}.png',
                                     run_id=logger.run_id)

        # 4.) distribution of max softmax predictions (considered obs only => passed threshold)
        idx_selected = np.where((outs_weak > threshold).sum(1) == 1)[0]
        counts_weak = (outs_weak > threshold).sum(0)
        counts_strong = (outs_strong > threshold).sum(0)
        y_selected = [y[idx] for idx in idx_selected]
        y_dist = np.unique(y_selected, return_counts=True)
        y_height = [int(y_dist[1][y_dist[0] == idx]) if idx in y_dist[0] else 0 for idx in range(len(y_all[0]))]

        fig, axs = plt.subplots(3)
        fig.suptitle(f"Predictive class distribution \nnon-annotated samples (>=threshold) in epoch {step}")

        axs[0].bar(x=np.arange(counts_weak.shape[0]), height=counts_weak)
        axs[0].set_xticks(np.arange(outs_weak.shape[1]))
        axs[0].set_title(f"Weakly augmented samples, selected {counts_weak.sum()}")

        axs[1].bar(x=np.arange(counts_strong.shape[0]), height=counts_strong)
        axs[1].set_xticks(np.arange(outs_strong.shape[1]))
        axs[1].set_title("Strongly augmented samples")

        axs[2].bar(x=np.arange(len(y_height)), height=y_height)
        axs[2].set_xticks(np.arange(outs_strong.shape[1]))
        axs[2].set_title("True labels")

        fig.tight_layout()

        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'pred_dist_thresh_{step}.png',
                                     run_id=logger.run_id)

        logger.experiment.log_metric(key='n_pseudo_labels',
                                     value=counts_weak.sum(),
                                     step=step,
                                     run_id=logger.run_id)
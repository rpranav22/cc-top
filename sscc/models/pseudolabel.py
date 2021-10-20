import torch.nn as nn
import torch
import pdb
import numpy as np
import pandas as pd

from torch.nn import functional as F

from sscc.losses import KCL, MCL
from sscc.metrics import Evaluator
from sscc.data.utils import Class2Simi, PairEnum


class PseudoLabel(nn.Module):
    """
    """
    def __init__(self, model, loss):
        super(PseudoLabel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = KCL().to(self.device) if loss == 'KCL' else MCL().to(self.device)

    def forward(self, batch, **kwargs):
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

    def evaluate(self, eval_dataloader, confusion=Evaluator(10), part='val'):
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
        return eval_results

    def pseudo_label(self, train_data, num_plabels):
        """
        """
        dataset = train_data
        c_df = dataset.c
        # delete old pseudo labels
        # we can potentially control for that => e.g. by if statement
        c_df = c_df[c_df.part == 'train']
        x = torch.tensor(dataset.x).to(torch.float64).to(self.device)
        y = torch.tensor(dataset.y).to(self.device)

        # 1.) get y_hat predictions
        y_hat = []
        for i in range(0, len(x), 64):
            x_batch = x[i:i+64, :, :, :]
            y_batch = self.forward(x_batch.float())
            y_hat.append(y_batch['out'].detach())
        y_hat = torch.cat(y_hat)

        # 2.) unconsider all obs data which we observe in the training data
        max_probs, max_class = torch.max(y_hat, dim=1)
        max_k_probs, max_k_indeces = torch.topk(max_probs, k=num_plabels)
        max_k_classes = max_class[max_k_indeces]

        # 3.) apply some pseudo-labeling logic
        df_pseudo = pd.DataFrame(columns=['idx', 'part', 'i', 'j', 'y_i', 'y_j', 'c_ij'])

        idx_cnt = len(c_df)
        # 4.) update constraints dataframe for training
        for (label_i, idx_i) in zip(max_k_classes, max_k_indeces):
            for (label_j, idx_j) in zip(max_k_classes, max_k_indeces):
                # no self constraints
                if idx_i == idx_j:
                    continue
                c_ij = -1 if label_j != label_i else 1
                temp = {'idx': idx_cnt,
                        'part': 'pseudo',
                        'i': int(idx_i),
                        'j': int(idx_j),
                        'y_i': int(label_i),
                        'y_j': int(label_j),
                        'c_ij': c_ij}
                df_pseudo = df_pseudo.append(temp, ignore_index=True)
                idx_cnt += 1
            # make sure that we do not double add symmetric constraints
            max_k_indeces = max_k_indeces[max_k_indeces != idx_i]
        # append pseudo constraints to our constraints train data set
        c_df = c_df.append(df_pseudo, ignore_index=True)
        dataset.c = c_df

        return dataset

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    def reset_model(self):
        """Reset the model weights, needed for training from scratch after each PL update strategy
        """
        print('I am re-setting my model weights to train again from scratch')
        self.model.apply(self.weight_reset)
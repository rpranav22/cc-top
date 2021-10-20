import pandas as pd
import numpy as np
import torch
import pdb
import os
import pytorch_lightning as pl
import warnings
import math
import tempfile
import yaml

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import optim
from torchvision.utils import save_image
from pytorch_lightning.loggers import MLFlowLogger

from sscc.data.utils import constrained_collate_fn, supervised_collate_fn, constraint_match_collate_fn, get_data
from sscc.data.images import ConstraintMatchData
from sscc.data.cifar10 import transforms_cifar10_weak, transforms_cifar10_strong
from sscc.data.yalebextend import transforms_yaleb_weak, transforms_yaleb_strong
from sscc.data.fashionmnist import transforms_fmnist_weak, transforms_fmnist_strong
from sscc.data.mnist import transforms_mnist_weak, transforms_mnist_strong
from sscc.metrics import Evaluator
from timeit import default_timer as timer

warnings.filterwarnings('ignore')

class Experiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 log_params,
                 run_name,
                 experiment_name,
                 trainer_params=None,
                 trial=None):
        super(Experiment, self).__init__()
        self.new_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.new_device)
        self.model.epoch = self.current_epoch
        self.params = params
        self.log_params = log_params
        self.trainer_params = trainer_params
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.trial = trial

        self.train_step = 0
        self.val_step = 0

        # initialize train/val/test data
        self.train_data = get_data(root='./data',
                                   dataset=self.params['dataset'],
                                   params=self.params,
                                   log_params=self.log_params,
                                   part='train')
        self.val_data = get_data(root='./data',
                                 dataset=self.params['dataset'],
                                 params=self.params,
                                 log_params=self.log_params,
                                 part='val')
        self.test_data = get_data(root='./data',
                                  dataset=self.params['dataset'],
                                  params=self.params,
                                  log_params=self.log_params,
                                  part='test')

    def _save_model_mlflow(self):
        """Save model outside of pl as checkpoint
        """        
        checkpoint = {'network': self.model, 
                      'state_dict': self.model.state_dict()}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'final_model.pt')
            with open(path, 'wb') as file:
                # store checkpoint in temp file
                torch.save(checkpoint, file)
                print(f'Stored final model')
                self.logger.experiment.log_artifact(local_path=path, run_id=self.logger.run_id)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch=batch)

        loss = self.model.loss_function(outs=outs, batch=batch, **self.params)

        for key, value in loss.items(): self.log(name=f"trainr_{key}", value=value, prog_bar=True)

        self.log(name='train_step', value=self.train_step)
        self.train_step += 1

        # pl selects the 'loss' element of the dict for backward
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0
        results = self.model.evaluate(eval_dataloader=self.train_gen, confusion=Evaluator(k=self.params['num_classes']), part='train')
        if self.current_epoch > 0: self.log_dict(results)
            # self.logger.experiment.log_metric(key=key,
            #                                   value=value,
            #                                   step=self.current_epoch,
            #                                   run_id=self.logger.run_id)

        # for the constraint match, we want to collect some stats on the pseudo-predictions for the unlabelled data
        if self.params['constraintmatch'] and (self.current_epoch + 1) % 2 and self.params['plot'] == 1:
            self.model.pl_stats(cm_train_gen=self.cm_train_gen,
                                threshold=self.params['threshold'],
                                logger=self.logger,
                                step=self.current_epoch)

        # here comes the pseudo labeling logic!
        # we need some logic after which num_epochs we want to have the pseudo_labels updated
        if self.params['pseudo_label']:
            if (self.current_epoch + 1) % self.params['pl_update'] == 0:
                self.train_data = self.model.pseudo_label(train_data=self.train_data, num_plabels=self.params['num_plabels'])
                # reset model after each pseudo label update
                self.model.reset_model()
        self.model.epoch = self.current_epoch

    def validation_step(self, batch, batch_idx):
        # batch as output by collate func in the data loader
        outs = self.forward(batch=batch)
        loss = self.model.loss_function(outs, batch, **self.params)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0

        # validation performance
        results = self.model.evaluate(eval_dataloader=self.val_gen,
                                      confusion=Evaluator(k=self.params['num_classes']),
                                      part='val',
                                      logger=self.logger,
                                      true_k=10 if self.params['dataset'] != 'cifar20' else 20)
        # skip epoch 0 as this is the sanity check of pt lightning
        if self.current_epoch > 0: self.log_dict(dictionary=results)

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        outs = self.forward(batch=batch)
        loss = self.model.loss_function(outs, batch, **self.params)

        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0

        scores = self.model.evaluate(eval_dataloader=self.test_dataloader(),
                                     confusion=Evaluator(k=self.params['num_classes']),
                                     part='test', 
                                     logger=self.logger,
                                     true_k=10 if self.params['dataset'] != 'cifar20' else 20)

        for key, value in zip(scores.keys(), scores.values()): self.log(name=key, value=value)

        self._save_model_mlflow()

        self.logger.experiment.log_param(key='run_name',
                                         value=self.run_name,
                                         run_id=self.logger.run_id)
        self.logger.experiment.log_param(key='experiment_name',
                                         value=self.experiment_name,
                                         run_id=self.logger.run_id)
        # log #constraints after TC/ CE
        self.logger.experiment.log_param(key=f"train_constraints_full",
                                          value=len(self.train_data.c),
                                          run_id=self.logger.run_id)
        self.logger.experiment.log_param(key=f"train_cl_ratio",
                                          value=round(np.unique(self.train_data.c['c_ij'], return_counts=True)[1][0] / sum(np.unique(self.train_data.c['c_ij'], return_counts=True)[1]), 4),
                                          run_id=self.logger.run_id)
        self.logger.experiment.log_param(key=f"train_ml_ratio",
                                          value=round(np.unique(self.train_data.c['c_ij'], return_counts=True)[1][1] / sum(np.unique(self.train_data.c['c_ij'], return_counts=True)[1]), 4),
                                          run_id=self.logger.run_id)
        self.logger.experiment.log_param(key='k', value=self.params['k'], run_id=self.logger.run_id)


    def configure_optimizers(self):
        optimizer = optim.SGD(params=self.model.parameters(),
                              lr=self.params['learning_rate'],
                              weight_decay=self.params['weight_decay'],
                              momentum=self.params['momentum'])
    
        if self.params['scheduler'] == 'cosine':
            # cosine scheduler as in Sohn 2020
            def lr_lambda(step: int) -> float:
                total_training_steps = int((self.trainer_params['max_epochs']) * (self.params['num_constraints'] / self.params['batch_size']))
                #total_training_steps = int(self.trainer_params['max_epochs'])
                factor = math.cos(7./ 16. * math.pi * (float(step) / float(max(1, total_training_steps))))
                # make sure it does not go < 0 due to underestimation of the total training steps
                #TODO: this is an ugly fix due to pls weird optimization schedule
                return max(factor, 0.000001)

            scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
            scheduler_config = {'scheduler': scheduler,
                                'interval': 'step'}

            return {'optimizer': optimizer, 'lr_scheduler': scheduler_config} 
        else:
            return {'optimizer': optimizer}

    def train_dataloader(self):

        self.train_gen = DataLoader(dataset=self.train_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=constrained_collate_fn,
                                    num_workers=self.params['num_workers'],
                                    shuffle=True)

        # if ConstraintMatch, apply DataLoader
        if self.params['constraintmatch']:
            if self.params['dataset'] == 'cifar10':
                transforms_weak = transforms_cifar10_weak
                transforms_strong = transforms_cifar10_strong
            if self.params['dataset'] == 'cifar20':
                transforms_weak = transforms_cifar10_weak
                transforms_strong = transforms_cifar10_strong
            elif self.params['dataset'] == 'yalebext':
                transforms_weak = transforms_yaleb_weak
                transforms_strong = transforms_yaleb_strong
            elif self.params['dataset'] == 'fashionmnist':
                transforms_weak = transforms_fmnist_weak
                transforms_strong = transforms_fmnist_strong
            elif self.params['dataset'] == 'mnist':
                transforms_weak = transforms_mnist_weak
                transforms_strong = transforms_mnist_strong

            cm_train_data = ConstraintMatchData(data=self.train_data,
                                                weak_transform=transforms_weak,
                                                strong_transform=transforms_strong)

            self.cm_train_gen = DataLoader(dataset=cm_train_data,
                                          batch_size=self.params['batch_size_ul'],
                                          collate_fn=constraint_match_collate_fn,
                                          num_workers=self.params['num_workers'],
                                          shuffle=True)

            return {'supervised_train': self.train_gen, 'cm_train': self.cm_train_gen}
        else:
            return self.train_gen

    def val_dataloader(self):

        self.val_gen = DataLoader(dataset=self.val_data,
                                  batch_size=self.params['batch_size'],
                                  collate_fn=supervised_collate_fn,
                                  num_workers=self.params['num_workers'],
                                  shuffle=True)

        return self.val_gen

    def test_dataloader(self):

        test_gen = DataLoader(dataset=self.test_data,
                              batch_size=self.params['batch_size'],
                              collate_fn=supervised_collate_fn,
                              num_workers=self.params['num_workers'],
                              shuffle=True)

        return test_gen

def save_dict_as_yaml_mlflow(data: dict, logger: MLFlowLogger, filename: str = 'config.yaml'): 
    """Store any dict in mlflow as an .yaml artifact

    Args:
        data (dict): input file, e.g. config 
        logger (MLFlowLogger): pytorch lightning mlflow logger (could be extended)
        filename (str): name for storage in mlflow artifacts
    """    
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
            logger.experiment.log_artifact(local_path=path, run_id=logger.run_id)


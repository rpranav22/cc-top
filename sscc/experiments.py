from functools import partial
import pandas as pd
import numpy as np
import torch
import pdb
import os
import pytorch_lightning as pl
import warnings
import tempfile
import yaml

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import optim
from torchvision.utils import save_image
from pytorch_lightning.loggers import MLFlowLogger

from sscc.data.utils import constrained_collate_fn, supervised_collate_fn, constraint_match_collate_fn, get_data, compute_metrics
from sscc.metrics import Evaluator

warnings.filterwarnings('ignore')

class Experiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 model_params,
                 log_params,
                 run_name,
                 experiment_name,
                 trainer_params=None,
                 trial=None):
        super(Experiment, self).__init__()
        self.new_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model_params = model_params
        # self.model.epoch = self.current_epoch
        self.params = params
        self.log_params = log_params
        self.trainer_params = trainer_params
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.trial = trial
        self.metric = compute_metrics

        self.train_step = 0
        self.val_step = 0

        if 'model_uri' in model_params:
            params['model_uri'] = model_params['model_uri']

        # initialize train/val/test data
        self.train_data = get_data(root='./data',
                                   params=self.params,
                                   log_params=self.log_params,
                                   part='train')
        self.val_data = get_data(root='./data',
                                 params=self.params,
                                 log_params=self.log_params,
                                 part='val')
        self.test_data = get_data(root='./data',
                                  params=self.params,
                                  log_params=self.log_params,
                                  part='test')
        print('loaded data cheers')

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
                print(f'Stored final model at {path}')
                self.logger.experiment.log_artifact(local_path=path, run_id=self.logger.run_id)

    def forward(self, *input, **kwargs):
            # print(type(input))
            # pdb.set_trace()
            return self.model(*input, **kwargs)

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        # fwd
        y_hat = self(input_ids, attention_mask, label)

        # loss
        loss = self.model.loss_function(out=y_hat, batch=batch, **self.params)

        for key, value in loss.items(): self.log(name=f"trainer_{key}", value=value, prog_bar=True)

        self.log(name='train_step', value=self.train_step)
        self.train_step += 1

        y_hat = y_hat.detach().cpu()
        batch['label'] = batch['label'].detach().cpu()

        return {'loss': loss['loss'], 'y_hat': y_hat, 'labels': batch['label']} #, 'train_batch': batch}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.detach()

        batch = [x['y_hat'] for x in outputs]

        results = self.model.evaluate(eval_dataloader=batch, confusion=Evaluator(k=self.model_params['architecture']['num_classes'])
                                        , labels=[x['labels'] for x in outputs]
                                        , current_epoch=self.current_epoch
                                        , part='train')
        if self.current_epoch > 0: self.log_dict(results)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids'] 
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss = self.model.loss_function(y_hat, batch, **self.params)


        self.log("val_loss", loss['loss'])

        return {'val_loss': loss , 'y_hat': y_hat, 'labels': batch['label']}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss']['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.detach()

        batch = [x['y_hat'] for x in outputs]
        # validation performance
        results = self.model.evaluate(eval_dataloader=batch,
                                      labels = [x['labels'] for x in outputs],
                                      confusion=Evaluator(k=self.model_params['architecture']['num_classes']),
                                      part='val',
                                      current_epoch=self.current_epoch,
                                      logger=self.logger,
                                      true_k=self.params['true_num_classes'])
        # skip epoch 0 as this is the sanity check of pt lightning
        if self.current_epoch > 0: self.log_dict(dictionary=results)

        # avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # print(f"\n\n___________________\n\naverage val accuracy is ---------> {avg_val_acc}\n\n")
        # self.log_dict({'avg_val_acc': avg_val_acc})


    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss = self.model.loss_function(y_hat, batch, **self.params)

        # a, y_hat = torch.max(y_hat, dim=1)
        # test_acc = self.metric(y_hat.cpu(), label.cpu())['accuracy']

        return {'test_loss': loss, 'y_hat': y_hat, 'labels': batch['label']}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss']['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.detach()

        scores = self.model.evaluate(eval_dataloader=[x['y_hat'] for x in outputs],
                                     labels=[x['labels'] for x in outputs],
                                     confusion=Evaluator(k=self.model_params['architecture']['num_classes']),
                                     part='test', 
                                     current_epoch=self.current_epoch,
                                     logger=self.logger,
                                     true_k=self.params['true_num_classes'])

        for key, value in zip(scores.keys(), scores.values()): self.log(name=key, value=value)

        self._save_model_mlflow()

        self.logger.experiment.log_param(key='run_name',
                                         value=self.run_name,
                                         run_id=self.logger.run_id)
        self.logger.experiment.log_param(key='experiment_name',
                                         value=self.experiment_name,
                                         run_id=self.logger.run_id)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.params['batch_size'] * max(1, self.trainer.gpus)
        # ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        ab_size = tb_size * self.trainer.accumulate_grad_batches

        self.total_steps = int((len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs))

        # total_devices = self.trainer.n_gpus * self.trainer.n_nodes
        # train_batches = len(self.train_dataloader()) // total_devices
        # self.train_steps = (self.trainer.epochs * train_batches) // self.trainer.accumulate_grad_batches
        print(f"estimated number of training steps is {self.total_steps} ")

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
    
        optimizer = optim.AdamW(model.parameters(), 
                            lr=self.params['learning_rate'], 
                            weight_decay=self.params['weight_decay'],
                            eps=self.params['adam_epsilon'])

       

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.params['warmup_steps'],
            # num_warmup_steps=2,
            num_training_steps=self.total_steps,
        )

        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        
        if self.params['constrained_clustering']:
            print(f'_____constrained____ \n')
            train_gen = DataLoader(dataset=self.train_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=partial(constrained_collate_fn, params=self.params),
                                    num_workers=self.params['num_workers'],
                                    shuffle=True)
            return train_gen
        else:
            train_gen = DataLoader(dataset=self.train_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=partial(supervised_collate_fn, params=self.params),
                                    num_workers=self.params['num_workers'],
                                    pin_memory=self.params['pin_memory'],
                                    shuffle=True)
            return train_gen

    def val_dataloader(self):

        val_gen = DataLoader(dataset=self.val_data,
                                  batch_size=self.params['batch_size'],
                                  collate_fn=partial(supervised_collate_fn, params=self.params),
                                  num_workers=self.params['num_workers'],
                                  pin_memory=self.params['pin_memory'],
                                  shuffle=True)

        return val_gen

    def test_dataloader(self):

        test_gen = DataLoader(dataset=self.test_data,
                              batch_size=self.params['batch_size'],
                              collate_fn=partial(supervised_collate_fn, params=self.params),
                              num_workers=self.params['num_workers'],
                              pin_memory=self.params['pin_memory'],
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


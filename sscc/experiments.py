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

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import optim
from torchvision.utils import save_image
from pytorch_lightning.loggers import MLFlowLogger

from sscc.data.utils import constrained_collate_fn, supervised_collate_fn, constraint_match_collate_fn, get_data, compute_metrics
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
        self.model = model
        self.model.epoch = self.current_epoch
        self.params = params
        self.log_params = log_params
        self.trainer_params = trainer_params
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.trial = trial
        self.metric = compute_metrics

        self.train_step = 0
        self.val_step = 0

        # initialize train/val/test data
        self.train_data = get_data(root='./data',
                                #    dataset=self.params['dataset'],
                                   params=self.params,
                                   log_params=self.log_params,
                                   part='train')
        self.val_data = get_data(root='./data',
                                #  dataset=self.params['dataset'],
                                 params=self.params,
                                 log_params=self.log_params,
                                 part='val')
        self.test_data = get_data(root='./data',
                                #   dataset=self.params['dataset'],
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

    def forward(self, **inputs):
            return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.params['num_classes'] >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.params['num_classes'] == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        # if self.hparams.task_name == "mnli":
        #     for i, output in enumerate(outputs):
        #         # matched or mismatched
        #         split = self.hparams.eval_splits[i].split("_")[-1]
        #         preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
        #         labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
        #         loss = torch.stack([x["loss"] for x in output]).mean()
        #         self.log(f"val_loss_{split}", loss, prog_bar=True)
        #         split_metrics = {
        #             f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
        #         }
        #         self.log_dict(split_metrics, prog_bar=True)
        #     return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric(preds, labels))
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.params['batch_size'] #* max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.params['learning_rate'], eps=self.params['adam_epsilon'])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.params['warmup_steps'],
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):

        self.train_gen = DataLoader(dataset=self.train_data,
                                    batch_size=self.params['batch_size'],
                                    # collate_fn=constrained_collate_fn,
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
                                #   collate_fn=supervised_collate_fn,
                                  num_workers=self.params['num_workers'],
                                  shuffle=True)

        return self.val_gen

    def test_dataloader(self):

        test_gen = DataLoader(dataset=self.test_data,
                              batch_size=self.params['batch_size'],
                            #   collate_fn=supervised_collate_fn,
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


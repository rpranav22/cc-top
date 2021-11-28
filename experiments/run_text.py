#!/usr/bin/env python
from logging import root
import yaml
import argparse
import torch 
# from transformers import Trainer, TrainingArguments
import numpy as np
import time
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
from sscc.data.newsgroups import newsgroups
from sscc.data.utils import get_data

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sscc.experiments import Experiment, save_dict_as_yaml_mlflow
from sscc.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='configs/newsgroup_supervised.yaml')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='amount of a priori classes')                        
    parser.add_argument('--batch_size', type=int, default=None,
                        help="batch size")                    
    parser.add_argument('--run_name', type=str, default=None,
                        help='name of the run')
    args = parser.parse_args()
    return args

def run_experiment(args):
    # torch.multiprocessing.set_start_method('spawn')
    # time.sleep(30)
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # update config
    config = update_config(config=config, args=args)
    params = config['exp_params']

    # compile model
    model = parse_model_config(config)

    # instantiate logger
    mlflow_logger = MLFlowLogger(experiment_name=config['logging_params']['experiment_name'])

    # for reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    # log all mlflow params
    for k, single_config in config.items():
        if k != 'search_space':
            mlflow_logger.log_hyperparams(params=single_config)

    # store the config
    save_dict_as_yaml_mlflow(data=config, logger=mlflow_logger)

    experiment = Experiment(model.model,
                            params=config['exp_params'],
                            log_params=config['logging_params'],
                            trainer_params=config['trainer_params'],
                            run_name=config['logging_params']['run_name'],
                            experiment_name=config['logging_params']['experiment_name'])

    # obtain data
    # train_data = get_data(root='./data', params=params, log_params=None, part='train')
    # val_data = get_data(root='./data', params=params, log_params=None, part='val')

    print(type(experiment.train_data), type(experiment.train_data.x))

    # model.run_lda(train_data.x)

    # training_args = TrainingArguments(
    #     output_dir='./results',          # output directory
    #     num_train_epochs=3,              # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=20,   # batch size for evaluation
    #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logger = mlflow_logger,            # directory for storing logs
    #     load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    #     # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    #     logging_steps=100,               # log & save weights each logging_steps
    #     evaluation_strategy="steps", 
    #      callbacks=[LearningRateMonitor(logging_interval='step')],
    #                  **config['trainer_params']    # evaluate each `logging_steps`
    # )
    
    trainer = Trainer(
        # model=model.model,                         # the instantiated Transformers model to be trained
        # args=training_args,                  # training arguments, defined above
        # train_dataset=train_data,         # training dataset
        # eval_dataset=val_data,          # evaluation dataset
        # compute_metrics=compute_metrics,     # the callback that computes metrics of interest
                    reload_dataloaders_every_epoch=False,
                    min_epochs=1,
                    log_every_n_steps=10,
                    checkpoint_callback=True,
                    logger=mlflow_logger,
                    check_val_every_n_epoch=1,
                    # num_sanity_val_steps=5,
                    # fast_dev_run=False,
                    # multiple_trainloader_mode='min_size',
                    callbacks=[LearningRateMonitor(logging_interval='step')],
                    **config['trainer_params']
    )

    trainer.fit(experiment)

    print("I have reached till here")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
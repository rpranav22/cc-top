#!/usr/bin/env python
from logging import root
import yaml
import argparse
import torch 
# from transformers import Trainer, TrainingArguments
import numpy as np
import time
import torch.backends.cudnn as cudnn
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
    mlflow_logger = MLFlowLogger(experiment_name=config['logging_params']['experiment_name'], run_name=config['logging_params']['run_name'])

    # for reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    # log all mlflow params
    for k, single_config in config.items():
        if k != 'search_space':
            mlflow_logger.log_hyperparams(params=single_config)

    # store the config
    save_dict_as_yaml_mlflow(data=config, logger=mlflow_logger)

    if config['model_params']['model'] == 'bertopic':
        print("not using lightning\n")
        train_data = get_data(root='./data', params=params, log_params=None, part='train')
        print(f"type: {type(train_data.x)}, length: {len(train_data.x)}")
        topic_model = model
        topics, probs = topic_model.fit_transform(train_data.x)
        print(f"done found {len(topics)} topics, they are \n\n")
        print(topic_model.get_topic_info())
        topic_model.save("bertopic_v1")
    
    else:
        print("Using LightningModule")

        experiment = Experiment(model,
                                params=config['exp_params'],
                                log_params=config['logging_params'],
                                trainer_params=config['trainer_params'],
                                run_name=config['logging_params']['run_name'],
                                experiment_name=config['logging_params']['experiment_name'])

        # obtain data
        # train_data = get_data(root='./data', params=params, log_params=None, part='train')
        # val_data = get_data(root='./data', params=params, log_params=None, part='val')

        print("Primer: ", len(experiment.train_data), type(experiment.train_data.x), len(experiment.train_data.c), len(experiment.test_data.y), len(experiment.test_data.c), len(experiment.val_data.y), len(experiment.val_data.c))
        print(f"model: {type(experiment.model)}")
        # model.run_lda(train_data.x)

        
        trainer = Trainer(
                        reload_dataloaders_every_epoch=False,
                        min_epochs=params['epochs'],
                        log_every_n_steps=100,
                        gpus=1,
                        checkpoint_callback=True,
                        logger=mlflow_logger,
                        check_val_every_n_epoch=1,
                        callbacks=[LearningRateMonitor(logging_interval='step')],
                        **config['trainer_params']
        )

        trainer.fit(experiment)
        trainer.test()

    print("I have reached till here")




if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
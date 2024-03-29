#!/usr/bin/env python
import os
import tempfile
import yaml
import argparse
import torch 
# from transformers import Trainer, TrainingArguments
import numpy as np
from cctop.data.utils import get_data

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from cctop.experiments import Experiment, save_dict_as_yaml_mlflow
from cctop.metrics import Evaluator
from cctop.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='configs/dbpedia_td_p1.yaml')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='amount of a priori classes')    
    parser.add_argument('--num_constraints', type=int, default=None,
                        help='amount of constraints')
    parser.add_argument('--test_set', type=str, default=None,
                        help='test data type')     
    parser.add_argument('--batch_size', type=int, default=None,
                        help="batch size")                  
    parser.add_argument('--max_epochs', type=int, default=None,
                        help="max epochs size")   
    parser.add_argument('--loss', type=str, default=None,
                        help="loss function")                   
    parser.add_argument('--run_name', type=str, default=None,
                        help='name of the run')
    parser.add_argument('--experiment_name', type=str,
                    help='name of the experiment')
    args = parser.parse_args()
    return args

def output_fn(p):
   p.export_chrome_trace("./trace/resnet50_4/worker0.pt.trace.json")

def run_experiment(args):
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
    
    elif config['model_params']['model'] == 'bert_kmeans':
        print("not lightning")
        train_data = get_data(root='./data', params=params, log_params=None, part='train')
        print(f"type: {type(train_data.x)}, length: {len(train_data.x)}")
        mlflow_logger.log_hyperparams({'num_samples':len(train_data.x)})
        cluster_assignments = model.fit_model(train_data.x)
        train_results = model.evaluate(batch=torch.tensor(cluster_assignments),
                                      labels = torch.tensor(train_data.y),
                                      confusion=Evaluator(k=config['model_params']['architecture']['num_classes']),
                                      part='train',
                                      logger=mlflow_logger,
                                      true_k=params['true_num_classes'])
        mlflow_logger.log_metrics(train_results)

        test_data = get_data(root='./data', params=params, log_params=None, part='test')
        print(f"type: {type(test_data.x)}, length: {len(test_data.x)}")
        test_cluster_assignments = model.predict(test_data.x)
        test_results = model.evaluate(batch=torch.tensor(test_cluster_assignments),
                                      labels = torch.tensor(test_data.y),
                                      confusion=Evaluator(k=config['model_params']['architecture']['num_classes']),
                                      part='test',
                                      logger=mlflow_logger,
                                      true_k=params['true_num_classes'])
        mlflow_logger.log_metrics(test_results)

    else:
        print("Using LightningModule")
        train_type = None
        if 'model_uri' in config['model_params']:
            phase = config['exp_params']['phase']
            print(f'topic discovery phase {phase} is happening')
            if config['exp_params']['train_type']:
                if config['exp_params']['train_type'] == 'finetune' or config['exp_params']['train_type'] == 'testing':
                    model_uri = config['model_params']['model_uri']
                    print(f"using model {model_uri} ")
                    model.load_state_dict(torch.load(model_uri)['state_dict'])
                    train_type = config['exp_params']['train_type']

        experiment = Experiment(model,
                                params=config['exp_params'],
                                model_params=config['model_params'], 
                                log_params=config['logging_params'],
                                trainer_params=config['trainer_params'],
                                run_name=config['logging_params']['run_name'],
                                experiment_name=config['logging_params']['experiment_name'])

        print(f"Primer:  train data size:{len(experiment.train_data)}, train_constraints: {len(experiment.train_data.c)}, test_size:{len(experiment.test_data.y)}, test_con: {len(experiment.test_data.c)}, val_size:{len(experiment.val_data.y)}, val_con: {len(experiment.val_data.c)}")
        print(f'sample data: {experiment.train_data.x[0]} \t label: {experiment.train_data.y[0]}')
        print(f"model: {type(experiment.model)}")

        constrained_samples = np.unique(experiment.train_data.c[['i', 'j']].values)
        num_samples = len(constrained_samples)
        print(f'\n\n Total no. of samples used for the constraints is {num_samples}')

        mlflow_logger.log_hyperparams({'total_samples': num_samples})
        with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, 'c_df_train.csv')
                experiment.train_data.c.to_csv(storage_path)
                mlflow_logger.experiment.log_artifact(local_path=storage_path, run_id=mlflow_logger.run_id)
        
        trainer = Trainer(
                        log_every_n_steps=100,
                        gpus=-1,
                        precision=16,
                        checkpoint_callback=True,
                        logger=mlflow_logger,
                        check_val_every_n_epoch=1,
                        callbacks=[LearningRateMonitor(logging_interval='step')],
                        **config['trainer_params'] 
        )

        if train_type == 'testing':
            trainer.test(experiment)
        else:
            trainer.fit(experiment)
            trainer.test()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
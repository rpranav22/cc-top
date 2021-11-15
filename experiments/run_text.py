#!/usr/bin/env python
import yaml
import argparse
import torch 
import numpy as np
import time
import torch.backends.cudnn as cudnn
from sscc.data.newsgroups import newsgroups

from sscc.experiments import Experiment, save_dict_as_yaml_mlflow
from sscc.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='configs/newsgroup_lda.yaml')
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

    if params['dataset'] == 'newsgroups':
        train_data = newsgroups(root='./data',
                       part='train',
                       val_size=params['val_size'],
                       num_constraints=params['num_constraints'],
                       k=params['k'])
        test_data = newsgroups(root='./data',
                       part='test',
                       val_size=params['val_size'],
                       num_constraints=params['num_constraints'],
                       k=params['k'])
        val_data = newsgroups(root='./data',
                       part='val',
                       val_size=params['val_size'],
                       num_constraints=params['num_constraints'],
                       k=params['k'])

    print(type(train_data), type(train_data.x), len(train_data.x))

    model.run_lda(train_data.x)

    model.evaluate()

    print("I have reached till here")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
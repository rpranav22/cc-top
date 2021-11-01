#!/usr/bin/env python
import os
import pdb
import torch
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from matplotlib import pyplot as plt
from sscc.data.utils import get_data
from sscc.utils import *

from sscc.data.cifar10 import transforms_cifar10_weak, transforms_cifar10_strong
from sscc.data.yalebextend import transforms_yaleb_weak, transforms_yaleb_strong
from sscc.data.fashionmnist import transforms_fmnist_weak, transforms_fmnist_strong

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='../../experiments/configs/yalebext_constraintmatch.yaml')
    parser.add_argument('--dataset', type=str, default='yalebext',
                        help="dataset to plot augs for")
    args = parser.parse_args()
    return args

# function to sample data
def make_data_sample(data_pool: np.ndarray = None, batch_size: int = 32):
    print('Sampling.')
    np.random.seed(1337)

    indices = np.random.randint(0, data_pool.shape[0], batch_size)
    example_batch = data_pool[indices]

    # define return variables
    x_aug_weak = np.empty_like(example_batch)
    x_aug_strong = np.empty_like(example_batch)

    # daug strategy
    if args.dataset == 'cifar10':
        transforms_weak = transforms_cifar10_weak
        transforms_strong = transforms_cifar10_strong
    elif args.dataset == 'yalebext':
        transforms_weak = transforms_yaleb_weak
        transforms_strong = transforms_yaleb_strong
    elif args.dataset == 'fashionmnist':
        transforms_weak = transforms_fmnist_weak
        transforms_strong = transforms_fmnist_strong

    print('Augmenting.')
    # apply daug
    for i in range(0, example_batch.shape[0]):
        x_aug_strong[i] = transforms_strong(torch.tensor(example_batch[i])).numpy()
        x_aug_weak[i] = transforms_weak(torch.tensor(example_batch[i])).numpy()

    return x_aug_strong, x_aug_weak

def data_saver(folder: str = 'daug_check', x_aug: np.ndarray = None, x_norm: np.ndarray = None):
    print('Saving.')
    # create the required folder
    Path(f'{folder}').mkdir(parents=True, exist_ok=True)
    num_images = x_aug.shape[0]

    for i in tqdm(range(0, num_images)):
        # first plotting efforts
        plt.close()
        fig = plt.figure()

        if x_norm.shape[1] == 1:
            vmin = x_norm[i].min()
            vmax = x_norm[i].max()

            plt.subplot(211)
            plt.imshow(x_norm[i].squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
            plt.title("Weak aug")
            plt.axis('off')

            plt.subplot(212)
            plt.imshow(x_aug[i].squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
            plt.title("Strong aug")
            plt.axis('off')
        else:
            plt.subplot(211)
            plt.imshow(np.transpose(x_norm[i], (1, 2, 0)))
            # plt.imshow((np.transpose(x_norm[i], (1, 2, 0))*255).astype(np.uint8))
            plt.title("Weak aug")
            plt.axis('off')

            plt.subplot(212)
            plt.imshow(np.transpose(x_aug[i], (1, 2, 0)))
            plt.title("Strong aug")
            plt.axis('off')

        plt.suptitle(f'sample{i}')

        plt.tight_layout()
        plt.savefig(f'{folder}/sample{i}.png', dpi=600)

def visualize_daug(args):
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = update_config(config, args)

    # get the data and other required params from config file
    data = get_data(root='../../experiments/data', dataset=config['exp_params']['dataset'],
                    params=config['exp_params'], log_params=config['logging_params'],
                    part='train')
    vis_batch_size = config['exp_params']['batch_size']

    x_aug, x_norm = make_data_sample(
        data_pool = data.x,
        batch_size = vis_batch_size)

    # save samples in folder
    data_saver('daug_check', x_aug, x_norm)

if __name__ == "__main__":
    args = parse_args()
    visualize_daug(args=args)

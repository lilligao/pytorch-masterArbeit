import torch
torch.set_float32_matmul_precision('high')

import argparse

## Parser Arguments
parser = argparse.ArgumentParser(description='EMUD Parser')

parser.add_argument('--project', type=str, default='EMUD (Default)')
parser.add_argument('--run', type=str, default='EMUD (Default)')
parser.add_argument('--backbone', type=str, default='b2')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--lr', type=float, default=6e-5)
parser.add_argument('--dataset', type=str, default='Cityscapes')
parser.add_argument('--use_scaling', type=bool, default=True)
parser.add_argument('--use_cropping', type=bool, default=True)
parser.add_argument('--use_flipping', type=bool, default=True)

args = parser.parse_args()

## W&B Logging (optional)
ENTITY = 'stevenlandgraf'   # change this to your W&B username
PROJECT = args.project
RUN_NAME = args.run

## Training & Hyperparameters
DEVICES = [0, 1, 2, 3]      # use 4 GPUs on the HPC. For local training or ipf server, just use one GPU.
NUM_EPOCHS = args.epochs
BACKBONE = args.backbone
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
PRECISION = '16-mixed'
LEARNING_RATE = args.lr

## Data Augmentation
USE_SCALING = args.use_scaling
USE_CROPPING = args.use_cropping
USE_FLIPPING = args.use_flipping

## Dataset & Dataloader
DATASET = args.dataset

if DATASET == 'Cityscapes':
    NUMBER_TRAIN_IMAGES = 2975
    NUMBER_VAL_IMAGES = 500
    BATCH_SIZE = 8
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    NUM_WORKERS = 8   

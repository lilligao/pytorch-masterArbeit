import torch
torch.set_float32_matmul_precision('high')
import argparse

## Parser Arguments
parser = argparse.ArgumentParser(description='Masterarbeit Segformer Parser')

parser.add_argument('--project', type=str, default='Masterarbeit Segformer')
parser.add_argument('--run', type=str, default='Segformer_train')
parser.add_argument('--backbone', type=str, default='b2') ## alle backbones ein mal probieren
parser.add_argument('--epochs', type=int, default=250)  # gpu: 250, local:1
parser.add_argument('--lr', type=float, default=6e-5) # gpu: default=6e-5, local:2e-1
parser.add_argument('--dataset', type=str, default='TLESS')
parser.add_argument('--root', type=str, default='./data/tless')
parser.add_argument('--train_split', type=str, default='train_pbr') # sensor daten und synthetic daten probieren
parser.add_argument('--val_split', type=str, default='train_pbr')
parser.add_argument('--test_split', type=str, default='test_primesense')
parser.add_argument('--val_size', type=str, default=10000)
parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--load_checkpoints', type=str, default=None)

parser.add_argument('--use_scaling', type=bool, default=True) #probieren verschiedene data augmentation
parser.add_argument('--use_cropping', type=bool, default=True) # vertikales flipping, rotation, farb sachen!!!
parser.add_argument('--use_flipping', type=bool, default=True)

args = parser.parse_args()

## W&B Logging (optional)
ENTITY = 'gaolilli'   # change this to your W&B username
PROJECT = args.project
RUN_NAME = args.run

## Training & Hyperparameters
#DEVICES = [0, 1, 2, 3]      # use 4 GPUs on the HPC. For local training or ipf server, just use one GPU.
DEVICES = 1    #cpu
NUM_EPOCHS = args.epochs
BACKBONE = args.backbone
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
PRECISION = 'bf16-mixed'   #gpu: '16-mixed', cpu: bf16-mixed
LEARNING_RATE = args.lr

## Data Augmentation
USE_SCALING = args.use_scaling
USE_CROPPING = args.use_cropping
USE_FLIPPING = args.use_flipping

## Dataset & Dataloader
DATASET = args.dataset
ROOT = args.root
TRAIN_SPLIT = args.train_split
VAL_SPLIT = args.val_split
TEST_SPLIT = args.test_split
VAL_SIZE = args.val_size
CHECKPOINTS_DIR = args.checkpoints
LOAD_CHECKPOINTS = args.load_checkpoints

if DATASET == 'TLESS':
    NUMBER_TRAIN_IMAGES = 50000 # 50000 total
    NUMBER_VAL_IMAGES = 10000  # 10080 total
    BATCH_SIZE = 8  # 8 for train on server with gpu, 2 for cpu
    NUM_CLASSES = 31 
    IGNORE_INDEX = None
    NUM_WORKERS = 8   

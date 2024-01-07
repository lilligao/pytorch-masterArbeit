import torch
torch.set_float32_matmul_precision('high')
import argparse

## Parser Arguments
parser = argparse.ArgumentParser(description='Masterarbeit Segformer Parser')

parser.add_argument('--project', type=str, default='Masterarbeit Segformer')
parser.add_argument('--run', type=str, default='Segformer_train')
parser.add_argument('--backbone', type=str, default='b5') ## alle backbones ein mal probieren
parser.add_argument('--epochs', type=int, default=250)  # gpu: 250, local:1
parser.add_argument('--lr', type=float, default=6e-5) # gpu: default=6e-5, local:2e-1
parser.add_argument('--lr_factor', type=int, default=1)
parser.add_argument('--dataset', type=str, default='TLESS')
parser.add_argument('--root', type=str, default='./data/tless')
parser.add_argument('--train_split', type=str, default='train_pbr;train_primesense') # sensor daten und synthetic daten probieren
parser.add_argument('--val_split', type=str, default='train_pbr')
parser.add_argument('--test_split', type=str, default='test_primesense')
parser.add_argument('--val_size', type=str, default=10000)
parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--load_checkpoints', type=str, default=None)
parser.add_argument('--plot_testimg', type=str, default='False')
parser.add_argument('--mAP_proImg', type=str, default='False')

parser.add_argument('--use_scaling', type=str, default='True') #probieren verschiedene data augmentation
parser.add_argument('--use_cropping', type=str, default='True') # vertikales flipping, rotation, farb sachen!!!
parser.add_argument('--use_flipping', type=str, default='True')
parser.add_argument('--k_intensity', type=int, default=0)
parser.add_argument('--use_normal_resize', type=str, default='False')
parser.add_argument('--scale_val', type=str, default='False')

parser.add_argument('--train_size', type=int, default=512)

parser.add_argument('--gradient_clip', type=float, default=0)
parser.add_argument('--gradient_clip_algorithm', type=str, default='norm')
parser.add_argument('--accumulate_grad_batches', type=int, default=1)

args = parser.parse_args()

## W&B Logging (optional)
ENTITY = 'gaolilli'   # change this to your W&B username
PROJECT = args.project
RUN_NAME = args.run

## Training & Hyperparameters
DEVICES = [0, 1, 2, 3]      # use 4 GPUs on the HPC. For local training or ipf server, just use one GPU.
#DEVICES = 1    #cpu
NUM_EPOCHS = args.epochs
BACKBONE = args.backbone
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
PRECISION = '16-mixed'   #gpu: '16-mixed', cpu: bf16-mixed
LEARNING_RATE = args.lr
LEARNING_RATE_FACTOR = args.lr_factor

## Data Augmentation
USE_SCALING = args.use_scaling
USE_CROPPING = args.use_cropping
USE_FLIPPING = args.use_flipping
K_INTENSITY = args.k_intensity
USE_NORMAL_RESIZE = args.use_normal_resize
SCALE_VAL = args.scale_val

## Dataset & Dataloader
DATASET = args.dataset
ROOT = args.root
TRAIN_SPLIT = args.train_split
VAL_SPLIT = args.val_split
TEST_SPLIT = args.test_split
VAL_SIZE = args.val_size

# test step
CHECKPOINTS_DIR = args.checkpoints
LOAD_CHECKPOINTS = args.load_checkpoints
PLOT_TESTIMG = args.plot_testimg
MAP_PROIMG = args.mAP_proImg

TRAIN_SIZE = args.train_size

GRADIENT_CLIP =args.gradient_clip
GRADIENT_CLIP_ALGORITHM = args.gradient_clip_algorithm
ACCUMULATE_GRAD_BATCHES = args.accumulate_grad_batches

if DATASET == 'TLESS':
    NUMBER_TRAIN_IMAGES = 50000 # 50000 total
    NUMBER_VAL_IMAGES = 10000  # 10080 total
    BATCH_SIZE = 8  # 8 for train on server with gpu, 2 for cpu
    NUM_CLASSES = 31 
    IGNORE_INDEX = None
    NUM_WORKERS = 8   

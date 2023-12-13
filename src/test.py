import lightning as L
from datasets.tless import TLESSDataset
import numpy as np
import matplotlib.pyplot as plt
import config
from models.segformer import SegFormer
from torch.utils.data import DataLoader
import train
import torch
from datasets.tless import TLESSDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from pathlib import Path
import glob

if __name__ == '__main__':
    #assert(config.LOAD_CHECKPOINTS!=None)
    chkpt_dir = "/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/checkpoints/Attempt5_31Classes/epochepoch=180-val_loss=0.05-val_iou=0.93.ckpt"
    model = SegFormer.load_from_checkpoint(chkpt_dir)

    data_module = TLESSDataModule(
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        root=config.ROOT,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT
    )

     # initialize the Trainer
    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='cpu',    # cpu
        strategy='auto',
        devices=1,
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        limit_train_batches=1.0, # or 0.25 for 25% # cpu:10
        limit_val_batches=1.0, # cpu:10
        max_steps=-1, # cpu:10
        #logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        callbacks=[
            ModelCheckpoint(dirpath=os.path.expandvars(config.CHECKPOINTS_DIR),filename='{config.RUN_NAME}-{epoch}-{val_loss:.2f}-{val_iou:.2f}',every_n_epochs=1, save_top_k= -1), 
            #ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'), # gewichte des Modells gespeichert nach bestimmter Epochen / beste Modell raus zu nehmen !! iteration nummer dran hängen
            LearningRateMonitor(logging_interval='epoch'),
        ],
        log_every_n_steps=1,
    )

    #test the model
    trainer.test(model, datamodule=data_module) 


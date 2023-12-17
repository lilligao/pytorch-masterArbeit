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
    assert(config.LOAD_CHECKPOINTS!=None)
    path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    chkpt = list(glob.glob(path))

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
        accelerator='gpu',    # cpu
        strategy='auto',
        devices=1,
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        #logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        callbacks=[
            ModelCheckpoint(dirpath=os.path.expandvars(config.CHECKPOINTS_DIR),filename='{config.RUN_NAME}-{epoch}-{val_loss:.2f}-{val_iou:.2f}',every_n_epochs=1, save_top_k= -1), 
            #ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'), # gewichte des Modells gespeichert nach bestimmter Epochen / beste Modell raus zu nehmen !! iteration nummer dran hängen
            LearningRateMonitor(logging_interval='epoch'),
        ],
        log_every_n_steps=1,
    )
    for i in range(len(chkpt)):
        print("checkpoint: ", chkpt[i])
        model = SegFormer.load_from_checkpoint(chkpt[i])
        #test the model
        trainer.test(model, datamodule=data_module,verbose=True) 


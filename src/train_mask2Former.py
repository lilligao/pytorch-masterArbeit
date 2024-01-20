import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import config
from models.mask2former import Mask2Former
from datasets.tless import TLESSMask2FormerDataModule
import os


if __name__ == '__main__':
    L.seed_everything(42)   # for reproducibility for training, aus welchen Seed, Zufaelligkeit raus, wichtig fuer data augmentation (random gewichte)
    if config.LOAD_CHECKPOINTS is not None:
        model = Mask2Former.load_from_checkpoint(config.LOAD_CHECKPOINTS)
    else:
        model = Mask2Former()

    data_module = TLESSMask2FormerDataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=1, #config.NUM_WORKERS
        root=config.ROOT,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
    )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='cpu',    # gpu
        strategy='auto',
        devices=1, #config.DEVICES
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        limit_train_batches=10, # or 0.25 for 25% # cpu:10
        limit_val_batches=10, # cpu:10
        max_steps=-1, # cpu:10
        #logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        callbacks=[
            ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}',filename='{epoch}-{val_loss:.2f}-{val_iou:.2f}',every_n_epochs=1, save_top_k=-1), 
            #ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'), # gewichte des Modells gespeichert nach bestimmter Epochen / beste Modell raus zu nehmen !! iteration nummer dran hängen
            LearningRateMonitor(logging_interval='epoch'),
        ],
        log_every_n_steps=1,
        max_time={"days": 1, "hours": 23},
        gradient_clip_val=config.GRADIENT_CLIP, # try other values!!!
        gradient_clip_algorithm=str(config.GRADIENT_CLIP_ALGORITHM),
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES
    )
    if config.LOAD_CHECKPOINTS is not None:
        trainer.fit(model, data_module,ckpt_path=config.LOAD_CHECKPOINTS) #ckpt_path = './checkpoints/name.clkpt'
    else:
        trainer.fit(model, data_module) #ckpt_path = './checkpoints/name.clkpt'

# lighting logs: tensorboard --logdir=lightning_logs/ --load_fast=false
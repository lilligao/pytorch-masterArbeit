import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import config
from models.segformer import SegFormer
from datasets.tless import TLESSDataModule


if __name__ == '__main__':
    L.seed_everything(42)   # for reproducibility for training, aus welchen Seed, Zufaelligkeit raus, wichtig fuer data augmentation (random gewichte)

    model = SegFormer()

    data_module = TLESSDataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        root=config.ROOT,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
    )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='cpu',    # gpu
        strategy='auto',
        devices=config.DEVICES,
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        limit_train_batches=10, # or 0.25 for 25% # cpu
        limit_val_batches=10, # cpu
        max_steps=10, # cpu
        #logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        callbacks=[
            ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'), # gewichte des Modells gespeichert nach bestimmter Epochen / beste Modell raus zu nehmen !! iteration nummer dran hängen
            LearningRateMonitor(logging_interval='epoch'),
        ],
        log_every_n_steps=1
    )

    trainer.fit(model, data_module)

# lighting logs: tensorboard --logdir=lightning_logs/ --load_fast=false
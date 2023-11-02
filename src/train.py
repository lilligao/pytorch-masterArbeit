import lightning as L

import config
from models.segformer import SegFormer
from datasets.tless import TLESSDataModule


if __name__ == '__main__':
    L.seed_everything(42)   # for reproducibility ???

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
        check_val_every_n_epoch=2,
        # logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        # logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        # callbacks=[
        #     ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'),
        #     LearningRateMonitor(logging_interval='epoch'),
        # ],
    )

    trainer.fit(model, data_module)
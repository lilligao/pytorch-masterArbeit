import lightning as L

import config
from models.segformer import SegFormer
from datasets.cityscapes import CityscapesDataModule


if __name__ == '__main__':
    L.seed_everything(42)   # for reproducibility

    model = SegFormer()

    data_module = CityscapesDataModule(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='gpu',
        strategy='auto',
        devices=config.DEVICES,
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        # logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=False),
        # logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        # callbacks=[
        #     ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}'),
        #     LearningRateMonitor(logging_interval='epoch'),
        # ],
    )

    trainer.fit(model, data_module)
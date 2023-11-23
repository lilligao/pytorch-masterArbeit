import math

import lightning as L
import torch
import torchmetrics
from transformers import SegformerForSemanticSegmentation, SegformerConfig

import config
from utils.lr_schedulers import PolyLR
import wandb

class SegFormer(L.LightningModule):
    def __init__(self):
        super().__init__()

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config)
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads imagenet weights

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def training_step(self, batch, batch_index):
        #images, _, labels = batch # if masks / masks visible are also in outpus
        images, labels = batch

        # print("train image shape",images.shape)
        # print("train label shape",labels.shape)
        loss, logits = self.model(images, labels.squeeze(dim=1))
        
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        self.train_iou(torch.softmax(upsampled_logits, dim=1), labels.squeeze(dim=1))

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # wandb.log({'batch': batch_index, 'loss': 0.3}) for log on step
        # gpu
        epoch = self.current_epoch
        step = self.global_step
        # wandb.log({'epoch': epoch, 'val_acc': 0.94}) for log on epoche

        if wandb.run is not None:
            wandb.log({'step': step, "train_loss_step": loss}) 
            wandb.log({'step': step,"train_iou_step": self.train_iou})
            wandb.log({'epoch': epoch, "train_loss_epoch": loss}) 
            wandb.log({'epoch': epoch,"train_iou_epoch": self.train_iou})
        
        return loss
    

    def validation_step(self, batch, batch_index):
        #images, _, labels = batch
        images, labels = batch

        print("evaluation image shape",images.shape)
        print("evaluation label shape",labels.shape)

        loss, logits = self.model(images, labels.squeeze(dim=1))
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        self.val_iou(torch.softmax(upsampled_logits, dim=1), labels.squeeze(dim=1))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        epoch = self.current_epoch
        step = self.global_step
        if wandb.run is not None:
            wandb.log({'step': step, "val_loss_step": loss}) 
            wandb.log({'step': step,"val_iou_step": self.val_iou})
            wandb.log({'epoch': epoch, "val_loss_epoch": loss}) 
            wandb.log({'epoch': epoch,"val_iou_epoch": self.val_iou})

    
    def configure_optimizers(self):
        # optimizer wird fuer jede Step gemacht, einmal über die Datensatz
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES))) # gpu
        #iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * config.DEVICES)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        print("iterations per epoche", iterations_per_epoch)
        print("total iterations", total_iterations)
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
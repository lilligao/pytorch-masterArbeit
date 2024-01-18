import math

import lightning as L
import torch
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerModel, MaskFormerImageProcessor

import config
from utils.lr_schedulers import PolyLR
import wandb
import datasets.tless as tless
class Mask2Former(L.LightningModule):
    def __init__(self):
        super().__init__()

        model_name = "facebook/mask2former-swin-base-coco-instance"
        # Get the MaskFormer config and print it
        config_mask2Former = Mask2FormerConfig.from_pretrained(model_name)
        dict(zip(range(1,31), range(1,31)))
        id2label = dict(zip(range(30), range(1,31)))
        label2id = {v: k for k, v in id2label.items()}
        # Edit MaskFormer config labels
        config_mask2Former.num_labels = 30
        config_mask2Former.id2label = id2label
        config_mask2Former.label2id = label2id
        config_mask2Former.return_dict = True
        config_mask2Former.ignore_index = 255
        #print("[INFO] displaying the MaskFormer configuration...")
        #print(config_mask2Former)
        

        # Use the config object to initialize a MaskFormer model with randomized weights
        self.model = Mask2FormerForUniversalSegmentation(config_mask2Former)
        # Replace the randomly initialized model with the pre-trained model weights
        base_model = Mask2FormerModel.from_pretrained(model_name)
        self.model.model = base_model

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.parameters(), 'lr': config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        # metrics for training
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX) #, ignore_index=config.IGNORE_INDEX
        self.train_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for validation
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for testing

        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        self.test_map = MeanAveragePrecision(iou_type="segm")
        self.test_map.compute_with_cache = False

        self.processor = MaskFormerImageProcessor(
            reduce_labels=True,
            size=(512, 512),
            ignore_index=255,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
                


    def training_step(self, batch, batch_index):
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        mask_labels=batch["mask_labels"]
        class_labels=batch["class_labels"]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]

        print("train image shape",pixel_values.shape)
        print("train pixel mask shape",pixel_mask.shape)
        print("train targets shape",target.shape)
        #print('train: ', torch.unique(labels.squeeze(dim=1)).tolist())

        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        
        loss = outputs.loss

        preds = self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_size)
        print("train preds length",len(preds))
        preds = torch.stack(preds)
        print("train preds shape",preds.shape)
        

        self.train_iou(preds, target)
        self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_mAP', self.train_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        mask_labels=batch["mask_labels"]
        class_labels=batch["class_labels"]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]

        print("val pixel_values shape",pixel_values.shape)
        print("val pixel_mask shape",pixel_mask.shape)
        print("val targets shape",target.shape)
        #print('val: ', torch.unique(labels.squeeze(dim=1)).tolist())

        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        
        loss = outputs.loss
        preds = self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_size)
        print("val preds length",len(preds))
        preds = torch.stack(preds)
        print("val preds shape",preds.shape)

        self.val_iou(preds, target)
        # self.val_ap(preds, target)
        #self.val_map.update(preds, target)


        # on epoche = True
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_ap', self.val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_mAP', self.val_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
     
    
    
    def configure_optimizers(self):
        # optimizer wird fuer jede Step gemacht, einmal über die Datensatz
        number_train_images = tless.NUMBER_TRAIN_IMAGES
        #iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * len(config.DEVICES))) # gpu
        iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * 1)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
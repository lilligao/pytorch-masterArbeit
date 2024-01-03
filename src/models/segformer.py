import math

import lightning as L
import torch
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision
from transformers import SegformerForSemanticSegmentation, SegformerConfig

import config
from utils.lr_schedulers import PolyLR
import wandb
import datasets.tless as tless
class SegFormer(L.LightningModule):
    def __init__(self):
        super().__init__()

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config)
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads imagenet weights

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': config.LEARNING_RATE_FACTOR * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        # metrics for training
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX) #, ignore_index=config.IGNORE_INDEX
        self.train_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro")
        # metrics for validation
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro")
        self.val_map = MeanAveragePrecision(iou_type="segm")
        # metrics for testing
        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro")
        self.test_map = MeanAveragePrecision(iou_type="segm")
         


    def training_step(self, batch, batch_index):
        #images, _, labels = batch # if masks / masks visible are also in outpus
        images, labels = batch

        # print("train image shape",images.shape)
        # print("train label shape",labels.shape)
        #print('train: ', torch.unique(labels.squeeze(dim=1)).tolist())
        target = labels.squeeze(dim=1)
        loss, logits = self.model(images, target)
        
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        self.train_iou(preds, target)
        self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_mAP', self.train_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        #images, _, labels = batch
        images, labels = batch

        #print("validation image shape",images.shape)
        #print("validation label shape",labels.shape)
        #print('valid: ', torch.unique(labels.squeeze(dim=1)).tolist())

        target = labels.squeeze(dim=1)
        loss, logits = self.model(images, target) # squeeze dim = 1 because labels size [4, 1, 540, 720]
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        self.val_iou(preds, target)
        self.val_ap(preds, target)
        #self.val_map.update(preds, target)

        # on epoche = True
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ap', self.val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_mAP', self.val_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # mean Average precision
        scores, preds = torch.max(preds, dim=1)# delete the first dimension
        preds = preds.squeeze(0) 
        scores = scores.squeeze(0)

        batch_size = preds.shape[0]

        preds_map = []
        targets_map = []

        for i in range(batch_size):
            # predictions
            preds_i = preds[i,:,:]
            scores_i = scores[i,:,:]
            detected_obj = torch.unique(preds_i).tolist()

            # targets
            target_i = target[i,:,:]
            target_obj = torch.unique(target_i).tolist()

            for j in detected_obj:
                mask_pred = preds_i==j
                score = torch.mean(scores_i[mask_pred]).item()
                preds_map.append(
                    dict(
                        masks=mask_pred.unsqueeze(0),
                        scores=torch.tensor([score]),
                        labels=torch.tensor([j]),
                    )
                )
                if j in target_obj:
                    mask_tgt = target_i==j
                    targets_map.append(
                        dict(
                            masks=mask_tgt.unsqueeze(0),
                            labels=torch.tensor([j]),
                        )
                    )
                else: # if something detected which is not in target, create a mask with all False
                    mask_tgt =  target_i==999
                    targets_map.append(
                        dict(
                            masks=mask_tgt.unsqueeze(0),
                            labels=torch.tensor([0]),
                        )
                    )

            for j in target_obj:
                if j not in detected_obj:
                    mask_tgt = target_i==j
                    targets_map.append(
                        dict(
                            masks=mask_tgt.unsqueeze(0),
                            labels=torch.tensor([j]),
                        )
                    )

                    mask_pred = preds_i==999
                    score = torch.mean(scores_i[mask_pred]).item()
                    preds_map.append(
                        dict(
                            masks=mask_pred.unsqueeze(0),
                            scores=torch.tensor([score]),
                            labels=torch.tensor([0]),
                        )
                    )
        self.val_map.update(preds=preds_map, target=targets_map)
        return self.val_map

        
    def on_validation_epoch_end(self, output):
        mAPs = output.compute()
        
        self.log_dict(mAPs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_map.reset()
        
    

    def test_step(self, batch, batch_idx):
        #images, _, labels = batch
        images, labels = batch

        print("test image shape",images.shape)
        print("test label shape",labels.shape)

        target = labels.squeeze(dim=1)
        loss, logits = self.model(images, target)
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        self.test_iou(preds, target)
        self.test_ap(preds, target)
        #self.test_map.update(preds, target)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_iou', self.test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_ap', self.test_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('test_mAP', self.test_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        ua = str("true").upper()
        if config.PLOT_TESTIMG.upper().startswith(ua):
            mask_data_tensor = torch.argmax(preds, dim=1).squeeze(0).cpu() # the maximum element
            mask_data = mask_data_tensor.numpy()
            mask_data_label_tensor =  labels.squeeze().cpu()
            mask_data_label = mask_data_label_tensor.numpy()
            class_labels = dict(zip(range(30), [str(i) for i in range(1,31)]))
            mask_img = wandb.Image(
                    images,
                    masks={
                        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
                        "ground_truth": {"mask_data": mask_data_label, "class_labels": class_labels},
                    },
                )
            if wandb.run is not None:
                # log images to W&B
                wandb.log({"predictions" : mask_img})

    
    def configure_optimizers(self):
        # optimizer wird fuer jede Step gemacht, einmal über die Datensatz
        number_train_images = tless.NUMBER_TRAIN_IMAGES
        iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * len(config.DEVICES))) # gpu
        #iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * config.DEVICES)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        print("iterations per epoche", iterations_per_epoch)
        print("total iterations", total_iterations)
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
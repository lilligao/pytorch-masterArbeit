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
        self.train_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for validation
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for testing

        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        self.test_map = MeanAveragePrecision(iou_type="segm")
        self.test_map.compute_with_cache = False
         


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
     
    

    def test_step(self, batch, batch_idx):
        #images, _, labels = batch
        images, labels = batch

        # print("test image shape",images.shape)
        # print("test label shape",labels.shape)

        target = labels.squeeze(dim=1)
        loss, logits = self.model(images, target)
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        self.test_iou(preds, target)
        self.log('test_iou', self.test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.test_ap(preds, target)
        self.log('test_ap', self.test_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.test_map.update(preds, target)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
         # mean Average precision
        scores, preds = torch.max(preds, dim=1)# delete the first dimension

        batch_size = preds.shape[0]

        preds_map = []
        targets_map = []

        for i in range(batch_size):
            scores_i = scores[i,:,:]
            # predictions ???? consider 0 in map oder niche????
            detected_obj = torch.unique(preds[i,:,:]).tolist()
            detected_obj.remove(0)
            
            # targets
            target_obj = torch.unique(target[i,:,:]).tolist()
            target_obj.remove(0)

            for j in detected_obj:
                mask_preds = preds[i,:,:]==j
                mask_tgt = target[i,:,:]==j if j in target_obj else target[i,:,:]==999 # if something detected which is not in target, create a mask with all False
                score = torch.mean(scores_i[mask_preds]).item()
                preds_map.append(
                    dict(
                        masks = mask_preds.unsqueeze(0),
                        scores=torch.tensor([score]),
                        labels=torch.tensor([j]),
                    )
                )
                targets_map.append(
                    dict(
                        masks = mask_tgt.unsqueeze(0),
                        labels=torch.tensor([j]),
                    )
                )

            for j in target_obj:
                if j not in detected_obj:
                    mask_tgt = target[i,:,:]==j
                    mask_preds = preds[i,:,:]==999
                    targets_map.append(
                        dict(
                            masks=mask_tgt.unsqueeze(0),
                            labels=torch.tensor([j]),
                        )
                    )

                    score = torch.mean(scores_i[mask_tgt]).item() # score of areas that exists obj in target
                    preds_map.append(
                        dict(
                            masks=mask_preds.unsqueeze(0),
                            scores=torch.tensor([score]),
                            labels=torch.tensor([j]), # the object j has mask of False
                        )
                    )
        # print("preds list", len(preds_map))
        # print("target list", len(targets_map))
        # print("preds mask", preds_map[1]["masks"].shape)
        # print("target mask", targets_map[1]["masks"].shape)
        self.test_map.update(preds=preds_map, target=targets_map)

        ua = str("true").upper()
        if config.MAP_PROIMG.upper().startswith(ua):
            # map
            mAPs = self.test_map.compute() #.to(self.device)
            mAPs.pop("classes")
            mAPs.pop("map_per_class")
            mAPs.pop("mar_100_per_class")
            self.log_dict(mAPs, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.test_map.reset()
            torch.cuda.empty_cache()

        
        if config.PLOT_TESTIMG.upper().startswith(ua):
            mask_data_tensor = preds.squeeze(0).cpu() # the maximum element
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
        
            
    def on_test_epoch_end(self):
        ua = str("true").upper()
        self.test_iou.reset()
        self.test_ap.reset()
        if not config.MAP_PROIMG.upper().startswith(ua):       
            mAPs = self.test_map.compute() #.to(self.device)
            mAPs.pop("classes")
            mAPs.pop("map_per_class")
            mAPs.pop("mar_100_per_class")
            self.log_dict(mAPs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.test_map.reset()
            torch.cuda.empty_cache()
    
    
    def configure_optimizers(self):
        # optimizer wird fuer jede Step gemacht, einmal über die Datensatz
        number_train_images = tless.NUMBER_TRAIN_IMAGES
        iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * len(config.DEVICES))) # gpu
        #iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * config.DEVICES)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
import math

import lightning as L
import torch
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrConfig, DetrForSegmentation, DetrImageProcessor

import config
from utils.lr_schedulers import PolyLR
import wandb
import datasets.tless as tless
class Detr(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the name of the model
        model_name = "facebook/detr-resnet-50-panoptic"
        config_detr = DetrConfig.from_pretrained(model_name)
        if config.NUM_CLASSES==30:
            id2label = dict(zip(range(30), range(1,31)))
        else:
            id2label = dict(zip(range(31), range(31)))
        label2id = {v: k for k, v in id2label.items()}
        # Edit MaskFormer config labels
        config_detr.num_labels = config.NUM_CLASSES
        config_detr.id2label = id2label
        config_detr.label2id = label2id
        config_detr.return_dict = True
        
        # Use the config object to initialize a MaskFormer model with randomized weights
        self.model = DetrForSegmentation(config_detr)

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.parameters(), 'lr': config.LEARNING_RATE_BACKBONE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        # metrics for training
        if config.NUM_CLASSES==30:
            self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX) #, ignore_index=config.IGNORE_INDEX
            self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
            self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        else:
            self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES) #, ignore_index=config.IGNORE_INDEX
            self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES)
            self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES)

        self.test_map = MeanAveragePrecision(iou_type="segm")
        self.test_map.compute_with_cache = False

        self.processor = DetrImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
         


    def training_step(self, batch, batch_index):
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]


        # print("train image shape",pixel_values.shape)
        # print("train pixel mask shape",pixel_mask.shape)
        # print("train targets shape",target.shape)
        #print('train: ', torch.unique(labels.squeeze(dim=1)).tolist())

        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels = labels
        )
    
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        preds = self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_size)
        # print("train preds length",len(preds))
        preds = torch.stack(preds)
        # print("train preds shape",preds.shape)
        

        self.train_iou(preds, target)
        #self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=list(target_shape)[0])
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())
        self.log('train_iou', self.train_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=list(target_shape)[0])
        return loss
    

    def validation_step(self, batch, batch_index):
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]

        # print("val pixel_values shape",pixel_values.shape)
        # print("val pixel_mask shape",pixel_mask.shape)
        # print("val targets shape",target.shape)
        #print('val: ', torch.unique(labels.squeeze(dim=1)).tolist())
        #print("label boxes",labels[0]["boxes"])

        # Forward pass
        if not self.trainer.sanity_checking:
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels = labels
            )
        
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            preds = self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_size)
            # print("val preds length",len(preds))
            preds = torch.stack(preds)
            # print("val preds shape",preds.shape)

            self.val_iou(preds, target)

            # on epoche = True
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=list(target_shape)[0])
            for k,v in loss_dict.items():
            self.log("val_" + k, v.item())
            self.log('val_iou', self.val_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=list(target_shape)[0])
        
        

    def test_step(self, batch, batch_idx):
        ua = str("true").upper()
        #images, _, labels = batch
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]

        print("test pixel_values shape",pixel_values.shape)
        print("test pixel_mask shape",pixel_mask.shape)
        print("test targets shape",target.shape)

        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels = labels
        )
    
        loss = outputs.loss
        preds = self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_size)
        
        preds = torch.stack(preds)

        self.test_iou(preds, target)
        self.log('test_iou', self.test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.test_ap(preds, target)
        #self.log('test_ap', self.test_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        #  # mean Average precision
        # scores, preds = torch.max(preds, dim=1)# delete the first dimension
        preds_dicts = self.processor.post_process_panoptic_segmentation(outputs,target_sizes=target_size) # !!! Threshold still to be determined
        batch_size = 2


        preds_map=[]
        targets_map=[]

        # plot the instance segmentation
        for i in range(batch_size):
            # masks and labels of target
            mask_tgt = labels[i]["masks"]
            label_tgt = labels[i]["class_labels"]
            mask_tgt = torch.as_tensor(mask_tgt, dtype=torch.uint8)
            print("label_tgt",label_tgt)
            print("target labels", label_tgt.shape)
            print("target masks", mask_tgt.shape)
            print("target masks", mask_tgt.dtype)
            print("target masks",torch.unique(mask_tgt))
            
            # masks and labels of prediction
            mask_preds = preds_dicts[i]["segmentation"]
            infos_preds = preds_dicts[i]["segments_info"]
            seg_preds = mask_preds

            scores = []
            labels = []
            masks = []
            for j in range(len(infos_preds)):
                segment_id =infos_preds[j]["id"]
                label_id = infos_preds[j]["label_id"]
                score_id = infos_preds[j]["score"]
                if config.NUM_CLASSES!=31 or label_id!=0:
                    mask_id =  mask_preds==segment_id
                    seg_preds[seg_preds==segment_id] = label_id
                    scores.append(score_id)
                    labels.append(label_id)
                    masks.append(mask_id)
            scores =  torch.as_tensor(scores, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.int)
            masks = torch.stack(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            print("preds score", scores.shape)
            print("preds labels", labels.shape)
            print("preds masks", masks.shape)
            print("preds masks", masks.dtype)
            print("preds masks",torch.unique(masks))
            print("preds masks",torch.unique(seg_preds))
            
            preds_map.append(
                        dict(
                            masks = masks,
                            scores=scores,
                            labels=labels,
                        )
                    )
            targets_map.append(
                        dict(
                            masks = mask_tgt,
                            labels= label_tgt,
                        )
                    )

            
            if config.PLOT_TESTIMG.upper().startswith(ua):
                # print("preds",seg_preds)
                # print("preds values",torch.unique(seg_preds))
                mask_data_tensor = seg_preds.cpu() # the maximum element
                mask_data = mask_data_tensor.numpy()
                mask_data_label_tensor =  target[i].squeeze().cpu()
                mask_data_label = mask_data_label_tensor.numpy()
                if config.NUM_CLASSES==30:
                    class_labels_dict = dict(zip(range(config.NUM_CLASSES), [str(i) for i in range(1,config.NUM_CLASSES+1)]))
                else:
                    class_labels_dict = dict(zip(range(config.NUM_CLASSES), [str(i) for i in range(config.NUM_CLASSES)]))
                mask_img = wandb.Image(
                        pixel_values[i],
                        masks={
                            "predictions": {"mask_data": mask_data, "class_labels": class_labels_dict},
                            "ground_truth": {"mask_data": mask_data_label, "class_labels": class_labels_dict},
                        },
                    )
                if wandb.run is not None:
                    # log images to W&B
                    wandb.log({"predictions" : mask_img})


                            
        print("preds list", len(preds_map))
        print("target list", len(targets_map))
        print("preds mask", preds_map[1]["masks"].shape)
        print("target mask", targets_map[1]["masks"].shape)
        self.test_map.update(preds=preds_map, target=targets_map)

        
        if config.MAP_PROIMG.upper().startswith(ua):
            # map
            mAPs = self.test_map.compute() #.to(self.device)
            mAPs.pop("classes")
            mAPs.pop("map_per_class")
            mAPs.pop("mar_100_per_class")
            self.log_dict(mAPs, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.test_map.reset()


            
    def on_test_epoch_end(self):
        ua = str("true").upper()
        self.test_iou.reset()
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
        optimizer = self.optimizer
        scheduler = PolyLR(optimizer, max_iterations=total_iterations, power=1.0)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
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
        if config.NUM_CLASSES==30:
            id2label = dict(zip(range(30), range(1,31)))
        else:
            id2label = dict(zip(range(31), range(31)))
        label2id = {v: k for k, v in id2label.items()}
        # Edit MaskFormer config labels
        config_mask2Former.num_labels = config.NUM_CLASSES
        config_mask2Former.id2label = id2label
        config_mask2Former.label2id = label2id
        config_mask2Former.return_dict = True
        config_mask2Former.ignore_index = config.IGNORE_INDEX
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

        ignore_index = int(config.IGNORE_INDEX) if config.IGNORE_INDEX is not None else None
        self.processor = MaskFormerImageProcessor(
            reduce_labels=False,
            ignore_index=ignore_index,
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

        # print("train image shape",pixel_values.shape)
        # print("train pixel mask shape",pixel_mask.shape)
        # print("train targets shape",target.shape)
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
        # print("train preds length",len(preds))
        preds = torch.stack(preds)
        # print("train preds shape",preds.shape)
        

        self.train_iou(preds, target)
        #self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        mask_labels=batch["mask_labels"]
        class_labels=batch["class_labels"]
        target = batch["target_segmentation"]
        target_shape = target.shape
        target_size = [list(target_shape)[1:]]*list(target_shape)[0]

        # print("val pixel_values shape",pixel_values.shape)
        # print("val pixel_mask shape",pixel_mask.shape)
        # print("val targets shape",target.shape)
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
        # print("val preds length",len(preds))
        preds = torch.stack(preds)
        # print("val preds shape",preds.shape)

        self.val_iou(preds, target)
        #self.val_ap(preds, target)
        #self.val_map.update(preds, target)


        # on epoche = True
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_ap', self.val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
     
    
    
    def test_step(self, batch, batch_idx):
        ua = str("true").upper()
        #images, _, labels = batch
        pixel_values=batch["pixel_values"]
        pixel_mask=batch["pixel_mask"]
        mask_labels=batch["mask_labels"]
        class_labels=batch["class_labels"]
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
            mask_labels=mask_labels,
            class_labels=class_labels,
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
        preds_dicts = self.processor.post_process_instance_segmentation(outputs,target_sizes=target_size) # !!! Threshold still to be determined
        batch_size = list(target_shape)[0]


        preds_map=[]
        targets_map=[]

        # plot the instance segmentation
        for i in range(batch_size):
            # masks and labels of target
            mask_tgt = mask_labels[i]
            label_tgt = class_labels[i].to(self.device)
            mask_tgt = torch.as_tensor(mask_tgt, dtype=torch.uint8).to(self.device)
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
                seg_preds[seg_preds==segment_id] = label_id
                print("Segment_id", segment_id)
                print("label_id", label_id)
                if config.NUM_CLASSES!=31 or label_id!=0:
                    mask_id =  mask_preds==segment_id
                    scores.append(score_id)
                    labels.append(label_id)
                    masks.append(mask_id)
            scores =  torch.as_tensor(scores, dtype=torch.float).to(self.device)
            labels = torch.as_tensor(labels, dtype=torch.int).to(self.device)
            if len(masks) >0:
                masks = torch.stack(masks)
                masks = torch.as_tensor(masks, dtype=torch.uint8).to(self.device)
            else:
                masks = torch.zeros_like(mask_tgt)

                print("preds masks", masks.shape)
                print("preds masks", masks.dtype)
                print("preds masks",torch.unique(masks))
                print("preds masks",torch.unique(seg_preds))
            
            print("preds score", scores.shape)
            print("preds labels", labels.shape)

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
        #iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * 1)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
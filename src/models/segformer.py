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
        if config.DROPOUT_RATE!=0.0:
            print("drop out rate:", config.DROPOUT_RATE)
            self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False,
                                                hidden_dropout_prob=config.DROPOUT_RATE,
                                                attention_probs_dropout_prob=config.DROPOUT_RATE,
                                                classifier_dropout_prob=config.DROPOUT_RATE, # default is 0.1
                                                drop_path_rate=config.DROPOUT_RATE,)  # default is 0.1
        else:
            print("Default dropout rate.")
            self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads imagenet weights

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': config.LEARNING_RATE_FACTOR * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        # metrics for training
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX) #, ignore_index=config.IGNORE_INDEX
        #self.train_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for validation
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        #self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        self.val_ece = torchmetrics.CalibrationError(task='multiclass', n_bins=10, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        # metrics for testing

        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ece = torchmetrics.CalibrationError(task='multiclass', n_bins=10, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
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
        #self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_mAP', self.train_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        #images, _, labels = batch
        images, labels = batch

        target = labels.squeeze(dim=1)
        loss, logits = self.model(images, target) # squeeze dim = 1 because labels size [4, 1, 540, 720]
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        self.val_iou(preds, target)
        #self.val_ap(preds, target)
        self.val_ece(preds, target)
        #self.val_map.update(preds, target)

        # on epoche = True
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ece', self.val_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_ap', self.val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_mAP', self.val_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
     
    def on_validation_epoch_end(self):
        self.val_iou.reset()
        self.val_ece.reset()
        # self.val_ap.reset()

    def test_step(self, batch, batch_idx):
         #images, _, labels = batch
        images, labels = batch
        if config.TEST_MODE=="MCDropout":
            ## Auskommentierte Sachen sind für MC-Dropout, das sollte man dann aber nicht während dem Training durchlaufen lassen, sondern im Anschluss, wenn man einen finalen Checkpoint hat
            # Activate dropout layers
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

            # For 5 samples
            sample_outputs = torch.empty(size=[config.NUM_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[-2], images.shape[-1]], device=self.device)
            for i in range(config.NUM_SAMPLES):
                loss, logits = self.model(images, labels.squeeze(dim=1))
                upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

                sample_outputs[i] = torch.softmax(upsampled_logits, dim=1)
                self.test_iou(sample_outputs[i], labels.squeeze(dim=1))
                #self.test_ece(sample_outputs[i],  labels.squeeze(dim=1))
            
            probability_map = torch.mean(sample_outputs, dim=0)
            prediction_map = torch.argmax(probability_map, dim=1, keepdim=True)
            standard_deviation_map = torch.std(sample_outputs, dim=0)
            entropy_map = torch.sum(-probability_map * torch.log(probability_map + 1e-6), dim=1, keepdim=True)

            # Beispiel für die Berechnung der Uncertainty Metrics mit der entropy_map. Analog könnte man es natürlich auch mit der standard_deviation_map machen.
            p_accurate_certain, p_inaccurate_uncertain, pavpu = self.compute_uncertainty_metrics(images, labels.squeeze(dim=1), prediction_map, entropy_map)
            p_accurate_certain_std, p_inaccurate_uncertain_std, pavpu_std = self.compute_uncertainty_metrics(images, labels.squeeze(dim=1), prediction_map, standard_deviation_map)
            self.log('pAccCer_entropy', p_accurate_certain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('pInaUnc_entropy', p_inaccurate_uncertain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('pavpu_entropy', pavpu, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.log('pAccCer_std', p_accurate_certain_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('pInaUnc_std', p_inaccurate_uncertain_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('pavpu_std', pavpu_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.log('test_iou', self.test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            #self.log('test_ece', self.test_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
           

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

                scores_preds = []
                labels_preds = []
                masks_preds = []
                for j in detected_obj:
                    
                    mask_preds = preds[i,:,:]==j
                    mask_tgt = target[i,:,:]==j if j in target_obj else target[i,:,:]==999 # if something detected which is not in target, create a mask with all False
                    score = torch.mean(scores_i[mask_preds]).item()

                    scores_preds.append(score)
                    labels_preds.append(j)
                    masks_preds.append(mask_preds)

                labels_tgt = []
                masks_tgt = []
                for j in target_obj:
                    mask_tgt = target[i,:,:]==j
                    labels_tgt.append(j)
                    masks_tgt.append(mask_tgt)
                
                scores_preds =  torch.as_tensor(scores_preds, dtype=torch.float)
                labels_preds = torch.as_tensor(labels_preds, dtype=torch.int)
                masks_preds = torch.stack(masks_preds)
                masks_preds = torch.as_tensor(masks_preds, dtype=torch.uint8)
                labels_tgt = torch.as_tensor(labels_tgt, dtype=torch.int)
                masks_tgt = torch.stack(masks_tgt)
                masks_tgt = torch.as_tensor(masks_tgt, dtype=torch.uint8)

                if config.PLOT_TESTIMG.upper().startswith(ua):
                    mask_data_tensor = preds.squeeze(0).cpu() # the maximum element
                    mask_data = mask_data_tensor.numpy()
                    mask_data_label_tensor =  labels[i].squeeze().cpu()
                    mask_data_label = mask_data_label_tensor.numpy()
                    class_labels = dict(zip(range(config.NUM_CLASSES), [str(i) for i in range(config.NUM_CLASSES)]))
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



                        score = torch.mean(scores_i[mask_tgt]).item() # score of areas that exists obj in target
                        preds_map.append(
                            dict(
                                masks=mask_preds.unsqueeze(0),
                                scores=torch.tensor([score]),
                                labels=torch.tensor([j]), # the object j has mask of False
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
                class_labels = dict(zip(range(config.NUM_CLASSES), [str(i) for i in range(config.NUM_CLASSES)]))
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
        if config.TEST_MODE!="MCDropout":
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
    
    def compute_uncertainty_metrics(self,images, labels, prediction, uncertainty):
        """
        Computes uncertainty metrics for a given set of images, labels, predictions, and uncertainties. Meant to be used for semantic segmentation.

        Args:
            images (torch.Tensor): A tensor of shape (N, C, H, W) representing the input images.
            labels (torch.Tensor): A tensor of shape (N, H, W) representing the ground-truth labels.
            prediction (torch.Tensor): A tensor of shape (N, H, W) representing the predicted labels.
            uncertainty (torch.Tensor): A tensor of shape (N, H, W) representing the uncertainty scores (predictive_uncertainty, entropy, standard deviation,...).

        Returns:
            Tuple of three floats representing the following metrics:
            - p_accurate_certain: The proportion of pixels that are accurate and certain.
            - p_uncertain_inaccurate: The proportion of pixels that are inaccurate and uncertain.
            - pavpu: The proportion of accurate pixels among the uncertain ones.
        """
        uncertainty_threshold = torch.mean(uncertainty) # Hier kannst du prinizipiell natürlich auch andere Thresholds dir überlegen. Ich habe hier einfach den Durchschnitt genommen. 

        binary_accuracy_map = (prediction == labels).float()

        # count the number of pixels that are accurate and certain
        n_ac = torch.sum((binary_accuracy_map == 1) & (uncertainty < uncertainty_threshold))

        # count the number of pixels that are accurate and uncertain
        n_au = torch.sum((binary_accuracy_map == 1) & (uncertainty >= uncertainty_threshold))

        # count the number of pixels that are inaccurate and certain
        n_ic = torch.sum((binary_accuracy_map == 0) & (uncertainty < uncertainty_threshold))

        # count the number of pixels that are inaccurate and uncertain
        n_iu = torch.sum((binary_accuracy_map == 0) & (uncertainty >= uncertainty_threshold))

        # compute the metrics
        p_accurate_certain = n_ac / (n_ac + n_ic)
        p_uncertain_inaccurate = n_iu / (n_ic + n_iu)
        pavpu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return p_accurate_certain, p_uncertain_inaccurate, pavpu
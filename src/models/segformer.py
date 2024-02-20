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
import numpy as np
import matplotlib.pyplot as plt
import os
class SegFormer(L.LightningModule):
    def __init__(self):
        super().__init__()

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config)
        if config.DROPOUT_RATE!=None:
            dropout_rate= float(config.DROPOUT_RATE)
            print("drop out rate:",dropout_rate)
            self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False,
                                                hidden_dropout_prob=dropout_rate,
                                                attention_probs_dropout_prob=dropout_rate,
                                                classifier_dropout_prob=dropout_rate, # default is 0.1
                                                drop_path_rate=dropout_rate,)  # default is 0.1
        else:
            print("Default dropout rate.")
            self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads imagenet weights

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': config.LEARNING_RATE_FACTOR * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # lightning: config optimizers -> scheduler anlegen!!!
        # metrics for training
        #self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX) #, ignore_index=config.IGNORE_INDEX
        #self.train_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        # metrics for validation
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        #self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        #self.val_ece = torchmetrics.CalibrationError(task='multiclass', n_bins=10, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        # metrics for testing

        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ece = torchmetrics.CalibrationError(task='multiclass', n_bins=10, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=config.NUM_CLASSES, average="macro",thresholds=100)
        self.test_map = MeanAveragePrecision(iou_type="segm")
        self.test_map.compute_with_cache = False
        self.images = []
        self.labels = []
        self.predictions = []
        self.dataset_entropy = []
        self.dataset_std = []
         


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

        #self.train_iou(preds, target)
        #self.train_ap(preds, target)
        #self.train_map.update(preds, target)

        # gpu:  on_step = False, on_epoch = True, cpu: on_step=True, on_epoch=False
        # !!! Metriken!!!
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
        #self.val_ece(preds, target)
        #self.val_map.update(preds, target)

        # on epoche = True
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_ece', self.val_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_ap', self.val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_mAP', self.val_map, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
         #images, _, labels = batch
        images, labels = batch #image: 1*3*540*720
        ua = str("true").upper()
        if config.TEST_MODE=="MCDropout":
            ## Auskommentierte Sachen sind für MC-Dropout, das sollte man dann aber nicht während dem Training durchlaufen lassen, sondern im Anschluss, wenn man einen finalen Checkpoint hat
            # Activate dropout layers
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

            # record time
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # For 5 samples
            sample_outputs = torch.empty(size=[config.NUM_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[-2], images.shape[-1]], device=self.device)
            for i in range(config.NUM_SAMPLES):
                loss, logits = self.model(images, labels.squeeze(dim=1))
                upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

                sample_outputs[i] = torch.softmax(upsampled_logits, dim=1)
            
            end.record()
            torch.cuda.synchronize()  # need to wait once more for op to finish
            
            probability_map = torch.mean(sample_outputs, dim=0)
            prediction_map = torch.argmax(probability_map, dim=1, keepdim=True) #1*1*540*720

            # Compute the predictive uncertainty
            standard_deviation_map = torch.std(sample_outputs, dim=0) #1*31*540*720
            predictive_uncertainty = torch.zeros(size=[images.shape[0], images.shape[2], images.shape[3]], device=self.device) # 1*540*720
            
            for i in range(config.NUM_CLASSES):
                #standard_deviation_map[:, i, :, :] 1*540*720
                predictive_uncertainty = torch.where(prediction_map.squeeze(0) == i, standard_deviation_map[:, i, :, :], predictive_uncertainty)

            entropy_map = torch.sum(-probability_map * torch.log(probability_map + 1e-6), dim=1, keepdim=True) #1*1*540*720
            self.test_iou(probability_map, labels.squeeze(dim=1))
            self.test_ece(probability_map,  labels.squeeze(dim=1))

            self.log('test_iou', self.test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('test_ece', self.test_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('sampling_time', start.elapsed_time(end), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

            self.images.append(images)
            self.labels.append(labels.squeeze(dim=1))
            self.predictions.append(prediction_map)
            self.dataset_std.append(predictive_uncertainty)
            self.dataset_entropy.append(entropy_map)

            
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
            self.test_ece(preds, target)
            self.log('test_ece', self.test_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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

                # print("masks_preds",masks_preds.shape)
                # print("labels_preds",labels_preds.shape)

                # print("masks_tgt",masks_tgt.shape)
                # print("labels_tgt",labels_tgt.shape)

                preds_map.append(
                    dict(
                        masks=masks_preds,
                        scores=scores_preds,
                        labels=labels_preds, # the object j has mask of False
                    )
                )
            
                targets_map.append(
                    dict(
                        masks=masks_tgt,
                        labels=labels_tgt, # the object j has mask of False
                    )
                )

                if config.PLOT_TESTIMG.upper().startswith(ua):
                    mask_data_tensor = preds.squeeze(0).cpu() # the maximum element
                    mask_data = mask_data_tensor.numpy()
                    mask_data_label_tensor =  labels[i].squeeze().cpu()
                    mask_data_label = mask_data_label_tensor.numpy()
                    class_labels = dict(zip(range(config.NUM_CLASSES), [str(p) for p in range(config.NUM_CLASSES)]))
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

            # print("preds list", len(preds_map))
            # print("target list", len(targets_map))
            # print("preds mask", preds_map[1]["masks"].shape)
            # print("target mask", targets_map[1]["masks"].shape)
            self.test_map.update(preds=preds_map, target=targets_map)

            if config.MAP_PROIMG.upper().startswith(ua):
                # map
                mAPs = self.test_map.compute() #.to(self.device)
                mAPs.pop("classes")
                mAPs.pop("map_per_class")
                mAPs.pop("mar_100_per_class")
                self.log_dict(mAPs, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.test_map.reset()
                torch.cuda.empty_cache()

        
            
    def on_test_epoch_end(self):
        ua = str("true").upper()
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
        else:
            entropy_dataset = []
            std_dataset = []
            mins_std_dataset = []
            maxs_std_dataset = []
            for i in range(len(self.images)):
                entropy_i = torch.mean(self.dataset_entropy[i])
                std_i = torch.mean(self.dataset_std[i])
                entropy_dataset.append(entropy_i)
                std_dataset.append(std_i)
                mins_std_dataset.append(torch.min(self.dataset_std[i]))
                maxs_std_dataset.append(torch.max(self.dataset_std[i]))

            mean_entropy_dataset = torch.mean(torch.stack(entropy_dataset))
            mean_std_dataset = torch.mean(torch.stack(std_dataset))
            min_std_dataset = torch.min(torch.stack(mins_std_dataset))
            max_std_dataset = torch.max(torch.stack(maxs_std_dataset))

            print("mean_entropy_dataset",mean_entropy_dataset)
            print("mean_std_dataset",mean_std_dataset)
            print("min_std_dataset",min_std_dataset)
            print("max_std_dataset",max_std_dataset)

            for i in range(len(self.images)):
                image_i = self.images[i]
                label_i = self.labels[i]
                prediction_i = self.predictions[i]
                uncertainty_std_i = self.dataset_std[i]
                uncertainty_entropy_i = self.dataset_entropy[i]
                # Beispiel für die Berechnung der Uncertainty Metrics mit der entropy_map. Analog könnte man es natürlich auch mit der standard_deviation_map machen.
                p_accurate_certain, p_inaccurate_uncertain, pavpu = self.compute_uncertainty_metrics(label_i, prediction_i, uncertainty_entropy_i, mean_entropy_dataset)
                p_accurate_certain_std, p_inaccurate_uncertain_std, pavpu_std = self.compute_uncertainty_metrics(label_i, prediction_i, uncertainty_std_i, mean_std_dataset)

                # binary map output
                binary_accuracy_map = (prediction_i == label_i).float()

                # min and max Std
                std_min_proImg = torch.min(uncertainty_std_i)
                std_max_proImg = torch.max(uncertainty_std_i)

                self.log('pAccCer_entropy', p_accurate_certain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('pInaUnc_entropy', p_inaccurate_uncertain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('pavpu_entropy', pavpu, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                # prove if on epoch is average!!!!!

                self.log('pAccCer_std', p_accurate_certain_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('pInaUnc_std', p_inaccurate_uncertain_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('pavpu_std', pavpu_std, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                self.log('std_min', min_std_dataset, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('std_max', max_std_dataset, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('std_mean', mean_std_dataset, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('entropy_mean', mean_entropy_dataset, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                if wandb.run is not None:
                    wandb.log({'step': i, "std_min_proImg": std_min_proImg}) 
                    wandb.log({'step': i, "std_max_proImg": std_max_proImg}) 

                #here didn't consider the situation when batch_size>0!
                if config.PLOT_TESTIMG.upper().startswith(ua):
                        detected_obj = torch.unique(prediction_i).tolist()
                        target_obj = torch.unique(label_i).tolist()

                        mask_data_tensor = prediction_i.squeeze().cpu() # the maximum element
                        mask_data = mask_data_tensor.numpy()
                        mask_data_label_tensor =  label_i.squeeze().cpu()
                        mask_data_label = mask_data_label_tensor.numpy()

                        mask_std = uncertainty_std_i.squeeze().cpu() 
                        mask_std = mask_std.numpy()
                        mask_std = (mask_std - min_std_dataset.cpu().numpy())/(max_std_dataset.cpu().numpy()-min_std_dataset.cpu().numpy()) # min und max werte ein loggen!!
                        mask_entropy = uncertainty_entropy_i.squeeze().cpu() 
                        mask_entropy = mask_entropy.numpy()

                        mask_binary = binary_accuracy_map.squeeze().cpu().numpy()

                        class_labels = dict(zip(range(config.NUM_CLASSES), [str(p) for p in range(config.NUM_CLASSES)]))
                        mask_img = wandb.Image(
                                image_i,
                                masks={
                                    "predictions": {"mask_data": mask_data, "class_labels": class_labels},
                                    "ground_truth": {"mask_data": mask_data_label, "class_labels": class_labels},
                                },
                            )
                        entropy_img = wandb.Image(mask_entropy)
                        std_img = wandb.Image(mask_std)
                        binary_img = wandb.Image(binary_accuracy_map)
                        list_outputs = [mask_img, entropy_img, std_img, binary_img]
                        if wandb.run is not None:
                            # log images to W&B
                            wandb.log({"predictions" : list_outputs})
                            # wandb.log({"predictions" : mask_img,
                            #            "std_img" : std_img,
                            #            "entropy_img" : entropy_img})
                        if config.PLOT_TESTIMG=='True_local':

                            directory = './outputs/'+ config.RUN_NAME +'/'
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            
                            # write detected obj ids to txt
                            with open(directory + str(i+1).zfill(4)+'_info.txt', 'w') as f:
                                f.write('obects in target label: ')
                                f.write(','.join(str(v) for v in target_obj) + '\n')
                                f.write('obects in prediction: ')
                                f.write(','.join(str(v) for v in detected_obj) + '\n')

                            # plot predicted segmentation mask
                            fig,ax = plt.subplots()
                            #print("img_array.shape",img_array.shape)
                            fig.frameon = False
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.set_size_inches(mask_data.shape[1]/100,mask_data.shape[0]/100)
                            fig.add_axes(ax)
                            ax.imshow(mask_data, cmap='gist_ncar', vmin=0, vmax=int(config.NUM_CLASSES)-1) # so for feste Klasse feste Farbe
                            fig.savefig(directory + str(i+1).zfill(4)+'_'+ 'predicted_label.png', dpi=100)
                            plt.close()

                            # plot target segmentation mask
                            fig,ax = plt.subplots()
                            fig.frameon = False
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.set_size_inches(mask_data_label.shape[1]/100,mask_data_label.shape[0]/100)
                            fig.add_axes(ax)
                            ax.imshow(mask_data_label, cmap='gist_ncar', vmin=0, vmax=int(config.NUM_CLASSES)-1) # so for feste Klasse feste Farbe
                            fig.savefig(directory + str(i+1).zfill(4)+'_'+ 'target_label.png', dpi=100)
                            plt.close()

                            # plot std 
                            fig,ax = plt.subplots()
                            fig.frameon = False
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.set_size_inches(mask_std.shape[1]/100,mask_std.shape[0]/100)
                            fig.add_axes(ax)
                            ax.imshow(mask_std,cmap='gray') # so for feste Klasse feste Farbe
                            fig.savefig(directory+ str(i+1).zfill(4)+'_'+ 'std.png', dpi=100)
                            plt.close()

                            # plot entropy 
                            fig,ax = plt.subplots()
                            fig.frameon = False
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.set_size_inches(mask_entropy.shape[1]/100,mask_entropy.shape[0]/100)
                            fig.add_axes(ax)
                            ax.imshow(mask_entropy,cmap='gray') # so for feste Klasse feste Farbe
                            fig.savefig(directory + str(i+1).zfill(4)+'_'+ 'entropy.png', dpi=100)
                            plt.close()

                            # plot binary map 
                            fig,ax = plt.subplots()
                            fig.frameon = False
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.set_size_inches(mask_binary.shape[1]/100,mask_binary.shape[0]/100)
                            fig.add_axes(ax)
                            ax.imshow(mask_binary,cmap='gray') # so for feste Klasse feste Farbe
                            fig.savefig(directory + str(i+1).zfill(4)+'_'+ 'binary_map.png', dpi=100)
                            plt.close()
    
    
    def configure_optimizers(self):
        # optimizer wird fuer jede Step gemacht, einmal über die Datensatz
        number_train_images = tless.NUMBER_TRAIN_IMAGES
        iterations_per_epoch = math.ceil(number_train_images / (config.BATCH_SIZE * len(config.DEVICES))) # gpu
        #iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * config.DEVICES)) # cpu
        total_iterations = iterations_per_epoch * self.trainer.max_epochs # for server with gpu
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=1.0)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def compute_uncertainty_metrics(self, labels, prediction, uncertainty, uc_threshold):
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
        uncertainty_threshold = uc_threshold # Hier kannst du prinizipiell natürlich auch andere Thresholds dir überlegen. Ich habe hier einfach den Durchschnitt genommen. 

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
from datasets.tless import TLESSDataset
import numpy as np
from models.segformer import SegFormer
import torch
import time
from itertools import groupby
from torch.utils.data import DataLoader
import json
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import AveragePrecision
from torchmetrics.detection import MeanAveragePrecision
import config
from torch import tensor
from pprint import pprint

def collate_fn(batch):
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1]["label"])
        return torch.stack(imgs), torch.stack(targets)

if __name__ == '__main__':
    # assert(config.LOAD_CHECKPOINTS!=None)
    # path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    # model = SegFormer.load_from_checkpoint(path)
    model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")

    dataloader =DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False, collate_fn=collate_fn)
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    
    metric_map = MeanAveragePrecision(iou_type="segm", max_detection_thresholds=[1,10,31])
    for data in dataloader:
        images, labels = data

        # print("test image shape",images.shape)
        # print("test label shape",labels.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)

        target = labels.squeeze(dim=1)
        loss, logits = model(images, target)
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        # mean Average precision
        scores, preds = torch.max(preds, dim=1)# delete the first dimension

        batch_size = preds.shape[0]

        preds_map = []
        targets_map = []

        for i in range(batch_size):
            # predictions
            detected_obj = torch.unique(preds[i,:,:]).tolist()

            # targets
            target_obj = torch.unique(target[i,:,:]).tolist()

            for j in detected_obj:
                score = torch.mean(scores[i,:,:][preds[i,:,:]==j]).item()
                preds_map.append(
                    dict(
                        masks = (preds[i,:,:]==j).unsqueeze(0),
                        scores=torch.tensor([score]),
                        labels=torch.tensor([j]),
                    )
                )
                if j in target_obj:
                    targets_map.append(
                        dict(
                            masks = (target[i,:,:]==j).unsqueeze(0),
                            labels=torch.tensor([j]),
                        )
                    )
                else: # if something detected which is not in target, create a mask with all False
                    targets_map.append(
                        dict(
                            masks = (target[i,:,:]==999).unsqueeze(0),
                            labels=torch.tensor([0]),
                        )
                    )

            for j in target_obj:
                if j not in detected_obj:
                    targets_map.append(
                        dict(
                            masks=(target[i,:,:]==j).unsqueeze(0),
                            labels=torch.tensor([j]),
                        )
                    )

                    score = torch.mean(scores[i,:,:][preds[i,:,:]==999]).item()
                    preds_map.append(
                        dict(
                            masks=(preds[i,:,:]==999).unsqueeze(0),
                            scores=torch.tensor([score]),
                            labels=torch.tensor([0]),
                        )
                    )
        print("preds list", len(preds_map))
        print("target list", len(targets_map))
        print("preds mask", preds_map[1]["masks"].shape)
        print("target mask", targets_map[1]["masks"].shape)
            
        #print(preds_map)
        metric_map.update(preds=preds_map, target=targets_map)
        # print("preds list", len(preds_map))
        # print("target list", len(targets_map))
        # print("preds mask", preds_map[1]["masks"].shape)
        # print("target mask", targets_map[1]["masks"].shape)
        pprint(metric_map.compute())
        #print("scene: " + str(target["scene_id"]) + ", image: " + str(target["image_id"]) + " done")

        
        del preds,logits
    







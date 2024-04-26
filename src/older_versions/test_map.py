import sys
 
# setting path
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/src/')
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
        scene_ids = []
        img_ids = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1]["label"])
            scene_ids.append(tensor(sample[1]["scene_id"]))
            img_ids.append(tensor(sample[1]["image_id"]))
        return torch.stack(imgs), torch.stack(targets), torch.stack(scene_ids), torch.stack(img_ids)

if __name__ == '__main__':
    # assert(config.LOAD_CHECKPOINTS!=None)
    # path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    # model = SegFormer.load_from_checkpoint(path)
    model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")

    dataloader =DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    
    metric_map = MeanAveragePrecision(iou_type="segm")
    for data in dataloader:
        images, labels, scene_ids, img_ids = data

        # print("test image shape",images.shape)
        # print("test label shape",labels.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)
        sample_outputs = torch.empty(size=[config.NUM_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[-2], images.shape[-1]], device=device)
        print("images shape", images.shape)
        #print("sample outputs", sample_outputs)
        print("sample outputs shape", sample_outputs.shape)

        target = labels.squeeze(dim=1)
        loss, logits = model(images, target)
    
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(upsampled_logits, dim=1)

        prediction_map = torch.argmax(preds, dim=1, keepdim=True)
        print("prediction_map shape", prediction_map.shape)
        # mean Average precision
        scores, preds = torch.max(preds, dim=1)# delete the first dimension
        print("prediction shape", preds.shape)
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
        print("scene: " + str(scene_ids) + ", image: " + str(img_ids) + " done")

        
        del preds,logits
    






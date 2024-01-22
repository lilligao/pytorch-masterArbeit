import sys
# setting path
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/src/')
import math
import os
import random
import matplotlib.pyplot as plt
import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerModel
from matplotlib.patches import Rectangle
import glob
import json

import config

from datasets.tless import TLESSDataset


def collate_fn(batch):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack([batch_i[0] for batch_i in batch])
    pixel_mask =torch.stack([batch_i[1]["pixel_mask"] for batch_i in batch])
    labels = []
    for batch_i in batch:
        new_target = {}
        mask_labels = batch_i[1]["mask_labels"]
        area_list = []
        for j in range(mask_labels.size(0)):
            area = mask_labels[j].sum()
            area_list.append(area)

        areas = torch.as_tensor(area_list,dtype=torch.float32)
        keep = areas>1
        new_target["image_id"] = batch_i[1]["scene_id"]*1000+batch_i[1]["image_id"]      
        new_target["class_labels"] = batch_i[1]["labels_detection"][keep] # delete masks that are not there after resize
        new_target["boxes"] = batch_i[1]["boxes"][keep] # resize / crop bounding box!
        new_target["area"] = areas[keep]
        new_target["iscrowd"] = torch.zeros_like(new_target["area"])
        new_target["masks"] = mask_labels[keep]
        labels.append(new_target)
    
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": torch.as_tensor(pixel_mask, dtype=torch.long),
        "labels": labels,
    }


if __name__ == '__main__':

    val_dataset = TLESSDataset(root='./data/tless', split='train_pbr',step="train")  #[0:10]
    #dataset = TLESSDataset(root='./data/tless', transforms=None, split='train_pbr')
    num_imgs = len(val_dataset)
    # img, target = val_dataset[0]    

    # label_array = target["label"].squeeze(0).numpy()
    # img = img.permute(1, 2, 0)
    # img_array = np.array(img)
    # print(img_array.shape)
    # #print(label_array)
    # print('label array:', label_array.shape)
    # print('target["mask_labels"] array:', target["mask_labels"].shape)
    # fig,ax = plt.subplots(1)
    # ax.imshow(img_array)
    # normal_boxes = target["boxes"].numpy()
    # for i in range(len(normal_boxes)):
    #     w= img.size(0)
    #     h= img.size(1)
        
    #     width = normal_boxes[i][2]*w
    #     height = normal_boxes[i][3]*h
    #     x= normal_boxes[i][0]*w - width/2
    #     y= normal_boxes[i][1]*h - height/2
    #     rect = Rectangle((x, y), width, height,edgecolor='orange', facecolor='none', linewidth=2)
    #     ax.add_patch(rect)
    #plt.show()
    dataloader =DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)
    for data in dataloader:
        break


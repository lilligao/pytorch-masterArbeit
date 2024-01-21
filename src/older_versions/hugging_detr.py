import sys
# setting path
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/src/')
import math
import os
import random

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
        new_target["image_id"] = batch_i[1]["scene_id"]*1000+batch_i[1]["image_id"]
        new_target["class_labels"] = batch_i[1]["labels_detection"] # delete masks that are not there after resize
        new_target["boxes"] = batch_i[1]["boxes"] # resize / crop bounding box!
        mask_labels = batch_i[1]["mask_labels"]
        area_list = []
        for j in range(mask_labels.size(0)):
            area = mask_labels[j].sum()
            area_list.append(area)

        new_target["area"] = torch.as_tensor(area_list,dtype=torch.float32)
        new_target["iscrowd"] = torch.zeros_like(new_target["area"])
        new_target["masks"] = batch_i[1]["mask_labels"]
        labels.append(new_target)

    # target_segmentation = torch.stack([batch_i[1]["label"].squeeze(0) for batch_i in batch])
    
    
    # class_labels = [batch_i[1]["labels_detection"] for batch_i in batch]
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": torch.as_tensor(pixel_mask, dtype=torch.long),
        "labels": labels,
    }


if __name__ == '__main__':

    val_dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="train")  #[0:10]
    #dataset = TLESSDataset(root='./data/tless', transforms=None, split='train_pbr')
    num_imgs = len(val_dataset)
    img, target = val_dataset[0]    

    dataloader =DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)

    for data in dataloader:
        t = data


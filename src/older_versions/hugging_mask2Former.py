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
    target_segmentation = torch.stack([batch_i[1]["label"].squeeze(0) for batch_i in batch])
    pixel_mask = target_segmentation!=config.IGNORE_INDEX
    mask_labels = [batch_i[1]["mask_labels"] for batch_i in batch]
    class_labels = [batch_i[1]["labels_detection"] for batch_i in batch]
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": torch.as_tensor(pixel_mask, dtype=torch.long),
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "target_segmentation": target_segmentation
    }



    

if __name__ == '__main__':

    val_dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="val")  #[0:10]
    #dataset = TLESSDataset(root='./data/tless', transforms=None, split='train_pbr')
    num_imgs = len(val_dataset)
    img, target = val_dataset[0]
    size = target["mask_labels"].shape
    print([list(size)[1:]]*list(size)[0] )
    print(torch.unique(target["mask_labels"]).tolist())
    print(target["mask_labels"].dtype)

    segmentation_map = target["label"].squeeze(0)
    print("segmentation map unique values", torch.unique(segmentation_map))
     # Get unique ids (class or instance ids based on input)
    ignore_index = config.IGNORE_INDEX
    instance_id_to_semantic_id = dict(zip(range(1,31), range(1,31)))
    reduce_labels = True
    

    # if reduce_labels:
    #     segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # # Get unique ids (class or instance ids based on input)
    # all_labels = np.unique(segmentation_map)
    # print(all_labels)
    # # Drop background label if applicable
    # if ignore_index is not None:
    #     all_labels = all_labels[all_labels != ignore_index]
    
    # print(all_labels)

    # # Generate a binary mask for each object instance
    # binary_masks = [(segmentation_map == i) for i in all_labels]
    # binary_masks = np.stack(binary_masks, axis=0)  # (num_labels, height, width)

    # # Convert instance ids to class ids
    # if instance_id_to_semantic_id is not None:
    #     labels = np.zeros(all_labels.shape[0])
        
    #     for label in all_labels:
    #         print(label)
    #         class_id = instance_id_to_semantic_id[label + 1 if reduce_labels else label]
    #         labels[all_labels == label] = class_id - 1 if reduce_labels else class_id
    # else:
    #     labels = all_labels

    # #return binary_masks.astype(np.float32), labels.astype(np.int64)
    # print("masks shape:",binary_masks.astype(np.float32).shape)
    # print("labels :",labels.astype(np.int64))

    dataloader =DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)
    
    # for data in dataloader:
    #     print(data["pixel_values"].shape)
    #     print(data["pixel_mask"].shape)
    #     print(len(data["mask_labels"]))
    #     print(len(data["class_labels"]))
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the name of the model
    model_name = "facebook/mask2former-swin-base-coco-instance"
    # Get the MaskFormer config and print it
    config_mask2Former = Mask2FormerConfig.from_pretrained(model_name)
    dict(zip(range(1,31), range(1,31)))
    id2label = dict(zip(range(30), range(1,31)))
    label2id = {v: k for k, v in id2label.items()}
    # Edit MaskFormer config labels
    config_mask2Former.num_labels = 30
    config_mask2Former.id2label = id2label
    config_mask2Former.label2id = label2id
    #config_mask2Former.return_dict = False
    print("[INFO] displaying the MaskFormer configuration...")
    print(config_mask2Former)
    

    # Use the config object to initialize a MaskFormer model with randomized weights
    model = Mask2FormerForUniversalSegmentation(config_mask2Former)
    # Replace the randomly initialized model with the pre-trained model weights
    base_model = Mask2FormerModel.from_pretrained(model_name)
    model.model = base_model
    #print(model)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # Set number of epochs and batch size
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} | Training")
        # Set model in training mode 
        model.train()
        train_loss, val_loss = [], []
        # Training loop
        for idx, batch in enumerate(dataloader):
            print("pixel_values",batch["pixel_values"].shape)
            print("pixel_mask",torch.unique(batch["pixel_mask"]))
            print("masks shape:",batch["mask_labels"][0][1].shape)
            print("labels shape:",batch["class_labels"][0].shape)
            print("mask values",torch.unique(batch["mask_labels"][0]).tolist())
            print(batch["mask_labels"][0].shape)
            print("class_labels",batch["class_labels"])
            # Reset the parameter gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            # Backward propagation
            loss = outputs.loss
            train_loss.append(loss.item())
            loss.backward()

            print("  Training loss: ", round(sum(train_loss)/len(train_loss), 6))
            # Optimization
            optimizer.step()
        # Average train epoch loss
        train_loss = sum(train_loss)/len(train_loss)
        # Set model in evaluation mode
        model.eval()
        start_idx = 0
        print(f"Epoch {epoch} | Validation")
        for idx, batch in enumerate(dataloader):
            with torch.no_grad():
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
                # Get validation loss
                loss = outputs.loss
                val_loss.append(loss.item())
                if idx % 50 == 0:
                    print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))
        # Average validation epoch loss
        val_loss = sum(val_loss)/len(val_loss)
        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")
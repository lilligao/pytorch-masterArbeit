# import sys
# sys.path.append( './src' )
import math
import os
import random

import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import glob
import json

import config


class TLESSDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, root, train_split, val_split):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.train_split = train_split
        self.val_split = val_split

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dataset = TLESSDataset(root=self.root, split=self.train_split) #[0:10]
        self.val_dataset = TLESSDataset(root=self.root, split=self.val_split)  #[0:10]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.batch_size / 2), shuffle=False, num_workers=self.num_workers, drop_last=False)

 
# TLESS dataset class for detector training
class TLESSDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split

        if self.split not in ['train_pbr','test_primesense', 'train_primesense','train_render_reconst']:
            raise ValueError(f'Invalid split: {self.split}')
        
        self.imgs = list(sorted(glob.glob(os.path.join(root, split, "*", "rgb",  "*.jpg" if split == 'train_pbr' else "*.png"))))
        self.depths = list(sorted(glob.glob(os.path.join(root, split, "*", "rgb",  "*.png"))))
        self.scene_gt_infos = list(sorted(glob.glob(os.path.join(root, split, "*", "scene_gt_info.json"))))
        self.scene_gts = list(sorted(glob.glob(os.path.join(root, split, "*", "scene_gt.json"))))

        self.ignore_index = config.IGNORE_INDEX
        self.void_classes = [0]
        self.valid_classes = range(1,31)     # classes: 30
        self.class_map = dict(zip(self.valid_classes, range(30)))
 

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        im_id = img_path.split('/')[-1].split('.')[0]
        scene_id = img_path.split('/')[-3]
 
        # Load mmage
        img = Image.open(img_path).convert("RGB")
 
        # Object ids
        with open(self.scene_gts[int(scene_id)]) as f:
            scene_gt = json.load(f)[str(int(im_id))]
        obj_ids = [gt['obj_id'] for gt in scene_gt]               
        
        # Load masks from mask
        masks_path = list(sorted(glob.glob(os.path.join(self.root, self.split, scene_id, "mask", f"{im_id}_*.png"))))
        masks = torch.zeros((len(masks_path), img.size[1], img.size[0]), dtype=torch.uint8)
        for i, mp in enumerate(masks_path):
            masks[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))
 
        #mask_visib
        masks_visib_path = list(sorted(glob.glob(os.path.join(self.root, self.split, scene_id, "mask_visib", f"{im_id}_*.png"))))
        masks_visib = torch.zeros((len(masks_path), img.size[1], img.size[0]), dtype=torch.uint8)
        for i, mp in enumerate(masks_visib_path):
            masks_visib[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))
 
        # create a single label image
        label = torch.zeros((img.size[1], img.size[0]), dtype=torch.int64)
        for i,id in enumerate(obj_ids):
            print(scene_id)
            print(id)
            print(id, np.sum(masks_visib[i].numpy()==255))
            label[masks_visib[i]==255] = id
        

        # Label Encoding
        for void_class in self.void_classes:
            label[label == void_class] = self.ignore_index
        for valid_class in self.valid_classes:
            label[label == valid_class] = self.class_map[valid_class]
        
        label = label.clone().detach().unsqueeze(0)     #torch.tensor(label, dtype=torch.long).unsqueeze(0)
 
 
        # Bounding boxes for each mask
        with open(self.scene_gt_infos[int(scene_id)]) as f:
            scene_gt_info = json.load(f)[str(int(im_id))]
        boxes = [gt['bbox_visib'] for gt in scene_gt_info]
        #print(boxes)
 
        img = TF.to_tensor(img)
        
        if self.split.startswith('train'):
            # Random Resize
            if config.USE_SCALING:
                random_scaler = RandResize(scale=(0.5, 2.0))
                img, masks, masks_visib, label = random_scaler(img.unsqueeze(0).float(), masks.unsqueeze(0).float(), masks_visib.unsqueeze(0).float(), label.unsqueeze(0).float())

                # Pad image if it's too small after the random resize
                if img.shape[1] < 512 or img.shape[2] < 512:
                    height, width = img.shape[1], img.shape[2]
                    pad_height = max(512 - height, 0)
                    pad_width = max(512 - width, 0)
                    pad_height_half = pad_height // 2
                    pad_width_half = pad_width // 2

                    border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                    img = F.pad(img, border, 'constant', 0)
                    masks = F.pad(masks, border, 'constant', 0)
                    masks_visib = F.pad(masks_visib, border, 'constant', 0)
                    label = F.pad(label, border, 'constant', self.ignore_index)

            # Random Horizontal Flip
            if config.USE_FLIPPING:
                if random.random() < 0.5:
                    img = TF.hflip(img)
                    masks = TF.hflip(masks)
                    masks_visib = TF.hflip(masks_visib)
                    label = TF.hflip(label)

            # Random Crop
            if config.USE_CROPPING:
                i, j, h, w = transforms.RandomCrop(size=(512, 512)).get_params(img, output_size=(512, 512))
                img = TF.crop(img, i, j, h, w)
                masks = TF.crop(masks, i, j, h, w)
                masks_visib = TF.crop(masks_visib, i, j, h, w)
                label = TF.crop(label, i, j, h, w) 
      

 
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels_detection"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["masks"] = masks
        target["masks_visib"] = masks_visib
        target["scene_id"] = torch.tensor([int(scene_id)])
        target["image_id"] = torch.tensor([int(im_id)])
        target["label"] = label # masken Bild
        
        return img, target["label"]  #target["label"]
    
    
class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, masks, masks_visib, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()
        
        temp_aspect_ratio = 1.0
        
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (
                self.aspect_ratio[0]
                + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            )
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)

        image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, size=(new_h, new_w), mode="bilinear", align_corners=False)
        masks_visib = F.interpolate(masks_visib, size=(new_h, new_w), mode="bilinear", align_corners=False)
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")

        return image.squeeze(), masks.squeeze(0), masks_visib.squeeze(0), label.squeeze(0).to(dtype=torch.int64)


def decode_segmentation_map(segmentation_map):
    """
    It takes a segmentation map, which is a 2D array of integers, and returns a 3D array of RGB values
    Example Usage: 
        image, label = dataset[0]
        output = model(image)
        decoded_output = decode_segmentation_map(output)
        decoded_label = decode_segmentation_map(label)
    """
    segmentation_map = segmentation_map.numpy()
    segmentation_map = segmentation_map.astype(np.uint8)
    img = Image.fromarray(segmentation_map)
    img_rgb = img.convert("RGB")
    rgb = np.array(img_rgb)
    rgb = TF.to_tensor(rgb)

    return rgb


# if __name__ == '__main__':
#     dataset = TLESSDataset(root='./data/tless', split='train_pbr')
#     num_imgs = len(dataset)
#     img, target = dataset[12]
#     unique_values = set()
#     for mask in target['masks']:
#         unique_values.update(torch.unique(mask).tolist())
    
#     image = TF.to_tensor(img)
#     print("iimage:", type(img))
#     print("label:", type(target["label"]))
#     print("mask:", type(target["masks"]))
#     print("mask_visib:", type(target["masks_visib"]))
#     print('num_imgs:',num_imgs)
#     print('labels: ', unique_values)
#     print('contains classes: ', torch.unique(target['labels_detection']).tolist())
#     print('test: ', torch.unique(target["label"]).tolist())
#     print('image:', img.shape)
#     print('label:', target["label"].shape)
    
#     label_array = target["label"].numpy()
#     img_array = np.array(img)
#     print(label_array)
#     print('label array:', label_array.shape)
    

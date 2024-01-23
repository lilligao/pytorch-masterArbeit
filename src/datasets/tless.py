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
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from transformers.image_transforms import corners_to_center_format

import glob
import json

import config
import sys
sys.path.append('./')
import lib.augs_TIBA as img_trsform

NUMBER_TRAIN_IMAGES = 0
class TLESSDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, root, train_split, val_split, test_split=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = os.path.expandvars(root)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def prepare_data(self):
        pass

    def setup(self, stage):
        splits = self.train_split.split(";")
        if self.train_split == self.val_split:
            train_set = []
            val_set = []
            for split_i in splits:
                #print(split_i)
                full_size_train = 0
                if split_i == "train_pbr": 
                    full_size_train = 50000 
                elif split_i == "train_primesense":
                    full_size_train = 37584 
                elif split_i == "train_render_reconst":
                    full_size_train = 76860 
                else:
                    raise ValueError(f'Invalid split: {split_i}')

                train_size = int(0.8 * full_size_train)
                val_size = full_size_train - train_size
                indexes = range(full_size_train)
                train_index, val_index = random_split(
                    dataset=indexes,
                    lengths=[train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                train_dataset = TLESSDataset(root=self.root, split=split_i,step="train", ind=train_index.indices)
                #print("training images:", len(train_dataset))
                val_dataset = TLESSDataset(root=self.root, split=split_i,step="val", ind=val_index.indices) 
                train_set.append(train_dataset)
                val_set.append(val_dataset)
            #print("training set length:", len(train_set))
            self.train_dataset = ConcatDataset(train_set)
            self.val_dataset =  ConcatDataset(val_set)
        else:
            train_set = []
            for split_i in splits:
                train_dataset = TLESSDataset(root=self.root, split=split_i,step="train") 
                #print("training images:", len(train_dataset))
                train_set.append(train_dataset)
            #print("training set length:", len(train_set))
            self.train_dataset = ConcatDataset(train_set)
            self.val_dataset = TLESSDataset(root=self.root, split=self.val_split,step="val") 
            
        if self.test_split is not None:
            self.test_dataset = TLESSDataset(root=self.root, split=self.test_split,step="test")  

        global NUMBER_TRAIN_IMAGES
        NUMBER_TRAIN_IMAGES = len(self.train_dataset)
        print("number of training images:", NUMBER_TRAIN_IMAGES)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.batch_size / 2), shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1]["label"])
        return torch.stack(imgs), torch.stack(targets)
       

# TLESS DataModule for Mask2Former
class TLESSMask2FormerDataModule(TLESSDataModule):
    def collate_fn(self, batch):
        # Get the pixel values, pixel mask, mask labels, and class labels
        pixel_values = torch.stack([batch_i[0] for batch_i in batch])
        target_segmentation = torch.stack([batch_i[1]["label"].squeeze(0) for batch_i in batch])
        pixel_mask =torch.stack([batch_i[1]["pixel_mask"] for batch_i in batch])
        mask_labels = [batch_i[1]["mask_labels"] for batch_i in batch]
        class_labels = [batch_i[1]["labels_detection"] for batch_i in batch]
        # Return a dictionary of all the collated features
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "target_segmentation": target_segmentation
        }


# TLESS DataModule for Detr
class TLESSDetrDataModule(TLESSDataModule):
    def collate_fn(self, batch):
         # Get the pixel values, pixel mask, mask labels, and class labels
        pixel_values = torch.stack([batch_i[0] for batch_i in batch])
        pixel_mask =torch.stack([batch_i[1]["pixel_mask"] for batch_i in batch])
        target_segmentation = torch.stack([batch_i[1]["label"].squeeze(0) for batch_i in batch])
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
            #new_target["image_id"] = batch_i[1]["scene_id"]*1000+batch_i[1]["image_id"]      
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
            "target_segmentation": target_segmentation
        }



# TLESS dataset class for detector training
class TLESSDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, step=None, ind=[]):
        self.root = os.path.expandvars(root)
        self.split = split
        self.step = step
        self.ind = ind

        if self.split not in ['train_pbr','test_primesense', 'train_primesense','train_render_reconst']:
            raise ValueError(f'Invalid split: {self.split}')
        
        self.imgs = list(sorted(glob.glob(os.path.join(self.root, split, "*", "rgb",  "*.jpg" if split == 'train_pbr' else "*.png"))))
        self.depths = list(sorted(glob.glob(os.path.join(self.root, split, "*", "depth",  "*.png"))))
        self.scene_gt_infos = list(sorted(glob.glob(os.path.join(self.root, split, "*", "scene_gt_info.json"))))
        self.scene_gts = list(sorted(glob.glob(os.path.join(self.root, split, "*", "scene_gt.json"))))

        if ind!=[]:
            self.imgs = [self.imgs[i] for i in ind]
            self.depths = [self.depths[i] for i in ind]

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

        #print("img id:",im_id)
        #print("scene_id",scene_id)
 
        # Load mmage
        img = Image.open(img_path).convert("RGB")
 
        # Object ids
        scene_gt_item = self.scene_gts[int(scene_id)] if self.split == 'train_pbr' else self.scene_gts[int(scene_id)-1]
        with open(scene_gt_item) as f:
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
            #print(id, np.sum(masks_visib[i].numpy()==255))
            label[masks_visib[i]==255] = id
        
        labels_detection = torch.as_tensor(obj_ids, dtype=torch.int64)

        # Label Encoding
        # void_classes: map the values in label 0-> 255
        if self.ignore_index is not None and config.NUM_CLASSES==30:
            for void_class in self.void_classes:
                label[label == void_class] = int(self.ignore_index)
            # map 1-30 -> 0-29
            for valid_class in self.valid_classes:
                label[label == valid_class] = self.class_map[valid_class]
                labels_detection[labels_detection == valid_class] = self.class_map[valid_class]
        
        label = label.clone().detach().unsqueeze(0)     #torch.tensor(label, dtype=torch.long).unsqueeze(0)
 
 
        # Bounding boxes for each mask
        scene_gt_infos_item = self.scene_gt_infos[int(scene_id)] if self.split == 'train_pbr' else self.scene_gt_infos[int(scene_id)-1]
        with open(scene_gt_infos_item) as f:
            scene_gt_info = json.load(f)[str(int(im_id))]
        boxes = np.asarray([gt['bbox_visib'] for gt in scene_gt_info])
        
        pixel_mask = torch.ones_like(img[0],dtype=torch.long)
        
        if self.step.startswith('train'):
            if config.K_INTENSITY > 0:
                transforms_list = [v2.RandomAutocontrast(p=1),
                                v2.RandomEqualize(p=1),
                                v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                                v2.RandomAdjustSharpness(p=1,sharpness_factor=2),
                                v2.RandomPosterize(p=1, bits=2),
                                #v2.RandomSolarize(p=1,threshold=200.0/255.0), # ??? sieht komisch aus
                                v2.ColorJitter(hue=.3),
                                v2.ColorJitter(brightness=.5), # in paper by [0.05,0.95]???
                                v2.ColorJitter(contrast=.5), # in paper by [0.05,0.95]???
                                v2.ColorJitter(saturation=.5), # in paper color balance???
                                ]
                transforms_list = random.sample(transforms_list,config.K_INTENSITY)
                random.shuffle(transforms_list)
                #print(transforms_list)
                transform_compose= transforms.Compose(transforms_list)
                if random.random() < 0.67:
                    img = transform_compose(img)
            elif config.K_INTENSITY < 0:
                strong_img_aug = img_trsform.strong_img_aug(abs(config.K_INTENSITY))
                img = strong_img_aug(img)
            
            img = TF.to_tensor(img)
            # Random Resize
            if str(config.USE_SCALING).upper()==str('True').upper():
                random_scaler = RandResize(scale=(0.5, 2.0))
                img, masks, masks_visib, label, boxes = random_scaler(img.unsqueeze(0).float(), masks.unsqueeze(0).float(), masks_visib.unsqueeze(0).float(), label.unsqueeze(0).float(), boxes)

                # Pad image if it's too small after the random resize
                if img.shape[1] < config.TRAIN_SIZE or img.shape[2] < config.TRAIN_SIZE:
                    height, width = img.shape[1], img.shape[2]
                    pad_height = max(config.TRAIN_SIZE - height, 0)
                    pad_width = max(config.TRAIN_SIZE - width, 0)
                    pad_height_half = pad_height // 2
                    pad_width_half = pad_width // 2

                    border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                    img = F.pad(img, border, 'constant', 0)
                    pixel_mask = F.pad(pixel_mask, border, 'constant', 0)
                    masks = F.pad(masks, border, 'constant', 0)
                    masks_visib = F.pad(masks_visib, border, 'constant', 0)
                    label = F.pad(label, border, 'constant', 0)
                    boxes[:, 0] += pad_width_half # add hegiht to top-left corner
                    boxes[:, 1] += pad_height_half

            # Random Horizontal Flip
            if str(config.USE_FLIPPING).upper()==str('True').upper():
                if random.random() < 0.5:
                    img = TF.hflip(img)
                    masks = TF.hflip(masks)
                    masks_visib = TF.hflip(masks_visib)
                    label = TF.hflip(label)
                    pixel_mask = TF.hflip(pixel_mask)
                    boxes[:, 0] = img.size(dim=2) - boxes[:, 0] -  boxes[:, 2] # img_width - x_topleft - width_bbox

            # Random Crop
            if str(config.USE_CROPPING).upper()==str('True').upper():
                i, j, h, w = transforms.RandomCrop(size=(config.TRAIN_SIZE, config.TRAIN_SIZE)).get_params(img, output_size=(config.TRAIN_SIZE, config.TRAIN_SIZE))
                img = TF.crop(img, i, j, h, w)
                masks = TF.crop(masks, i, j, h, w)
                masks_visib = TF.crop(masks_visib, i, j, h, w)
                label = TF.crop(label, i, j, h, w) 
                pixel_mask = TF.crop(pixel_mask, i, j, h, w) 
                boxes[:, 0] = boxes[:, 0]- j
                boxes[:, 1] = boxes[:, 1]- i
                
            # Normal resize
            if str(config.USE_NORMAL_RESIZE).upper()==str('True').upper():
                tf_img = transforms.Resize((config.TRAIN_SIZE, config.TRAIN_SIZE), interpolation=InterpolationMode.BILINEAR)
                tf = transforms.Resize((config.TRAIN_SIZE, config.TRAIN_SIZE), interpolation=InterpolationMode.NEAREST)
                img = tf_img(img)
                masks = tf(masks)
                masks_visib = tf(masks_visib)
                label = tf(label)
                pixel_mask = tf(pixel_mask)
                scale_factor_w = config.TRAIN_SIZE / img.size(dim=2)
                scale_factor_h = config.TRAIN_SIZE/ img.size(dim=1)
                boxes = boxes * np.asarray([scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h], dtype=np.float32)

        elif self.step.startswith('val') and str(config.SCALE_VAL).upper()==str('True').upper():
            img = TF.to_tensor(img)
            tf_img = transforms.Resize((config.TRAIN_SIZE, config.TRAIN_SIZE), interpolation=InterpolationMode.BILINEAR)
            tf = transforms.Resize((config.TRAIN_SIZE, config.TRAIN_SIZE), interpolation=InterpolationMode.NEAREST)
            img = tf_img(img)
            masks = tf(masks)
            masks_visib = tf(masks_visib)
            label = tf(label)
            pixel_mask = tf(pixel_mask)
            boxes = boxes * np.asarray([scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h], dtype=np.float32)

        # guard against no boxes via resizing
        image_height = img.size(dim=1)
        image_width =  img.size(dim=2)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # add hegiht to top-left corner
        boxes[:, 0::2].clamp_(min=0, max=image_width) # column 0 and 2
        boxes[:, 1::2].clamp_(min=0, max=image_height) # column 1 and 3
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        labels_detection = labels_detection[keep]
        masks = masks[keep]
        masks_visib = masks_visib[keep]

        boxes = corners_to_center_format(boxes) # top left bottom  to center of the box and its the width
        boxes /= torch.as_tensor([config.TRAIN_SIZE, config.TRAIN_SIZE, config.TRAIN_SIZE, config.TRAIN_SIZE], dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels_detection"] = labels_detection
        target["masks"] = masks
        target["masks_visib"] = masks_visib
        target["scene_id"] = int(scene_id)
        target["image_id"] = int(im_id)
        target["label"] = label # masken Bild
        target["mask_labels"] = torch.as_tensor(masks_visib==255, dtype=torch.float32)
        target["pixel_mask"] = pixel_mask # masken Bild
        
        return img, target  #target["label"]
    
    
class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, masks, masks_visib, label, boxes):
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
        masks = F.interpolate(masks, size=(new_h, new_w),mode="nearest")
        masks_visib = F.interpolate(masks_visib, size=(new_h, new_w), mode="nearest")
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")
        scaled_boxes = boxes * np.asarray([scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h], dtype=np.float32)

        return image.squeeze(), masks.squeeze(0), masks_visib.squeeze(0), label.squeeze(0).to(dtype=torch.int64), scaled_boxes


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

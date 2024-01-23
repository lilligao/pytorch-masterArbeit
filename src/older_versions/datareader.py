import os
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, v2
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from transformers import SegformerForSemanticSegmentation, SegformerConfig

import math
import os
import random

import lightning as L
from PIL import Image
from torch.nn import functional as F

import sys
 
# setting path
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/src/')
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/')

import glob
import json
import config

import lib.AugSeg.augseg.dataset.augs_TIBA as img_trsform

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
        n_valid = 200
        indexes = range(50000)
        train_index, val_index = random_split(
            dataset=indexes,
            lengths=[len(indexes)-n_valid, n_valid],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_dataset = TLESSDataset(root=self.root, split=self.train_split,step="train", ind=train_index.indices) 
        self.val_dataset = TLESSDataset(root=self.root, split=self.val_split,step="val", ind= val_index.indices)  
        if self.test_split is not None:
            self.test_dataset = TLESSDataset(root=self.root, split=self.test_split,step="test")  
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.batch_size / 2), shuffle=False, num_workers=self.num_workers, drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, num_workers=self.num_workers, drop_last=False)


 
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

        self.ignore_index = 0
        self.void_classes = [0]
        self.valid_classes = range(1,31)     # classes: 30
        self.class_map = dict(zip(self.valid_classes, range(30)))
 

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        model = SegformerForSemanticSegmentation(model_config)
        #print(model_config)
        #print(model)
        
        model = model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False,
                                      hidden_dropout_prob=0.2,
                                                attention_probs_dropout_prob=0.2,
                                                classifier_dropout_prob=0.2, # default is 0.1
                                                drop_path_rate=0.2,)
        for m in model.modules():
            print(m)
            print("------------------------------------------")

                                                

        img_path = self.imgs[idx]
        im_id = img_path.split('/')[-1].split('.')[0]
        scene_id = img_path.split('/')[-3]

        #print("img id:",im_id)
        #print("scene_id",scene_id)
        ignore_index = int(config.IGNORE_INDEX) if config.IGNORE_INDEX is not None else None
        print(ignore_index)
 
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
        

        # Label Encoding
        # void_classes: map the values in label 0-> 255
        # for void_class in self.void_classes:
        #     label[label == void_class] = self.ignore_index
        # # map 1-30 -> 0-29
        # for valid_class in self.valid_classes:
        #     label[label == valid_class] = self.class_map[valid_class]
        
        label = label.clone().detach().unsqueeze(0)     #torch.tensor(label, dtype=torch.long).unsqueeze(0)
 
 
        # Bounding boxes for each mask
        scene_gt_infos_item = self.scene_gt_infos[int(scene_id)] if self.split == 'train_pbr' else self.scene_gt_infos[int(scene_id)-1]
        with open(scene_gt_infos_item) as f:
            scene_gt_info = json.load(f)[str(int(im_id))]
        boxes = [gt['bbox_visib'] for gt in scene_gt_info]
        #print(boxes)
 
        #img = TF.to_tensor(img)
        

        if self.step.startswith('train'):
            print('contains classes before: ', torch.unique(label).tolist())
            # Random Resize
            if config.K_INTENSITY > 0:
                strong_img_aug = img_trsform.strong_img_aug(config.K_INTENSITY)
                img = strong_img_aug(img)
                # transforms_list = [v2.RandomAutocontrast(p=1), # normalize or maximize??
                #                 v2.RandomEqualize(p=1),
                #                 v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                #                 v2.RandomAdjustSharpness(p=1,sharpness_factor=2),
                #                 v2.RandomPosterize(p=1, bits=2),
                #                 #v2.RandomSolarize(p=1,threshold=200.0/255.0), # ??? sieht komisch aus
                #                 v2.ColorJitter(hue=.3),
                #                 v2.ColorJitter(brightness=.5), # in paper by [0.05,0.95]???
                #                 v2.ColorJitter(contrast=.5), # in paper by [0.05,0.95]???
                #                 v2.ColorJitter(saturation=.5), # in paper color balance???
                #                 ]
                # transforms_list = random.sample(transforms_list,config.K_INTENSITY)
                # random.shuffle(transforms_list)
                # print(transforms_list)
                # transform_compose= v2.Compose(transforms_list)
                # #transform_compose=transforms_list[8]
                # #print("min", torch.min(img[1,:,:]))
                # #print("max", torch.max(img[1,:,:]))
                # if random.random() < 0.67:
                #     img = transform_compose(img)
                #print("min", torch.min(img[1,:,:]))
                #print("max", torch.max(img[1,:,:]))
            
            img = TF.to_tensor(img)

            if True:
                random_scaler = RandResize(scale=(0.5, 0.9))
                
                img, masks, masks_visib, label = random_scaler(img.unsqueeze(0).float(), masks.unsqueeze(0).float(), masks_visib.unsqueeze(0).float(), label.unsqueeze(0).float())
                
                # Pad image if it's too small after the random resize
                if img.shape[1] < 512 or img.shape[2] < 512:
                    height, width = img.shape[1], img.shape[2]
                    pad_height = max(512 - height, 0)
                    pad_width = max(512 - width, 0)
                    pad_height_half = pad_height // 2
                    pad_width_half = pad_width // 2
                    pixel_mask = torch.ones_like(img[0],dtype=torch.long)
                    print("pixel mask shape",pixel_mask.shape)

                    border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                    img = F.pad(img, border, 'constant', 0)
                    pixel_mask = F.pad(pixel_mask, border, 'constant', 0)
                    print("pixel mask shape after padding",pixel_mask.shape)
                    print("pixel mask sum",torch.sum(pixel_mask))
                    masks = F.pad(masks, border, 'constant', 0)
                    masks_visib = F.pad(masks_visib, border, 'constant', 0)
                    label = F.pad(label, border, 'constant', self.ignore_index)
                    
            print('contains classes after padding: ', torch.unique(label).tolist())
            
            # Random Horizontal Flip
            if True:
                if random.random() < 0.5:
                    img = TF.hflip(img)
                    masks = TF.hflip(masks)
                    masks_visib = TF.hflip(masks_visib)
                    label = TF.hflip(label)
                    pixel_mask = TF.hflip(pixel_mask)
            print('contains classes after flipping: ', torch.unique(label).tolist())
            # Random Crop
            if str(config.USE_CROPPING).upper()==str('True').upper():
                print("use cropping!!")
                i, j, h, w = transforms.RandomCrop(size=(512, 512)).get_params(img, output_size=(512, 512))
                img = TF.crop(img, i, j, h, w)
                masks = TF.crop(masks, i, j, h, w)
                masks_visib = TF.crop(masks_visib, i, j, h, w)
                label = TF.crop(label, i, j, h, w) 
                pixel_mask = TF.crop(pixel_mask, i, j, h, w) 
            print('contains classes after random cropping: ', torch.unique(label).tolist())
        elif self.step.startswith('val'):
                tf_img = transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR)
                tf = transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST)
                img = tf_img(img)
                masks = tf(masks)
                masks_visib = tf(masks_visib)
                label = tf(label)
                pixel_mask = torch.ones_like(img[0],dtype=torch.long)
      
      
        print("pixel mask shape after padding",pixel_mask.shape)
        print("pixel mask sum",torch.sum(pixel_mask))
        print("img[0] shape", img[0].shape)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels_detection"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["masks"] = masks
        target["masks_visib"] = masks_visib
        target["scene_id"] = torch.tensor([int(scene_id)])
        target["image_id"] = torch.tensor([int(im_id)])
        target["label"] = label # masken Bild
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
        masks = F.interpolate(masks, size=(new_h, new_w), mode="nearest")
        masks_visib = F.interpolate(masks_visib, size=(new_h, new_w), mode="nearest")
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")

        return image.squeeze(), masks.squeeze(0), masks_visib.squeeze(0), label.squeeze(0).to(dtype=torch.int64)
        
 
if __name__ == '__main__':

    n_valid = 200
    indexes = range(50000)
    train_index, val_index = random_split(
        dataset=indexes,
        lengths=[len(indexes)-n_valid, n_valid],
        generator=torch.Generator().manual_seed(0)
    )
    #print("train:",train_index.indices)
    #print("val:",val_index.indices)
    train_dataset = TLESSDataset(root='./data/tless', split='train_pbr',step="train", ind=train_index.indices) #[0:10]
    val_dataset = TLESSDataset(root='./data/tless', split='train_primesense',step="val")  #[0:10]
    test_dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test") 

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    #dataset = TLESSDataset(root='./data/tless', transforms=None, split='train_pbr')
    num_imgs_train = len(train_dataset)
    num_imgs = len(val_dataset)
    num_imgs_test = len(test_dataset)
    img, target = train_dataset[1]
    # for i in range(0,num_imgs_train,200):
    #     img, target = train_dataset[i]
    #     a = torch.unique(target["label"]).tolist()
    #     print('test: ', torch.unique(target["label"]).tolist())
    #     assert(len(a)<31 and max(a)<31 and min(a)>=0)
    # unique_values = set()
    
    image = img
    print("iimage:", type(image))
    print("label:", type(target["label"]))
    print("mask:", type(target["masks"]))
    print("mask_visib:", type(target["masks_visib"]))
    print('num_imgs:',num_imgs_train)
    print('num_imgs:',num_imgs)
    print('num_imgs test dataset:',num_imgs_test)
    print('num_imgs test dataloader:',len(test_dataloader))
    
    print('contains classes: ', torch.unique(target['labels_detection']).tolist())
    
    print('image:', image.shape)

    unique_values = set()
    for mask in target["masks"]:
        unique_values.update(torch.unique(mask).tolist())
    
    print('labels: ', unique_values)
    
    label_array = target["label"].squeeze(0).numpy()
    img = image.permute(1, 2, 0)
    img_array = np.array(img)
    #print(label_array)
    print('label array:', label_array.shape)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(img_array)
    ax[1].imshow(label_array)
    plt.savefig('data/tless/label_img.png')
    plt.close()

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
import os
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# TLESS dataset class for detector training
class TLESSDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, split):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.imgs = list(sorted(glob.glob(os.path.join(root, split, "*", "rgb",  "*.jpg" if split == 'train_pbr' else "*.png"))))
        # self.masks = list(sorted(glob.glob(os.path.join(root, split, "*", "mask_visib", "*.png"))))
        self.scene_gt_infos = list(sorted(glob.glob(os.path.join(root, split, "*", "scene_gt_info.json"))))
        self.scene_gts = list(sorted(glob.glob(os.path.join(root, split, "*", "scene_gt.json"))))
 
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        im_id = img_path.split('/')[-1].split('.')[0]
        scene_id = img_path.split('/')[-3]

        print("img path:",img_path)
        print("img id:",im_id)
        print("scene_id",scene_id)

        # Load mmage
        img = Image.open(img_path).convert("RGB")
 
        # Object ids
        scene_gt_item = self.scene_gts[int(scene_id)] if self.split == 'train_pbr' else self.scene_gts[int(scene_id)-1]
        with open(scene_gt_item) as f:
            print("f:",f)
            scene_gt = json.load(f)[str(int(im_id))]
        obj_ids = [gt['obj_id'] for gt in scene_gt]    
        print("obj_ids", obj_ids)           
        
        # Load masks from mask
        masks_path = list(sorted(glob.glob(os.path.join(self.root, self.split, scene_id, "mask", f"{im_id}_*.png"))))
        masks = torch.zeros((len(masks_path), img.size[1], img.size[0]), dtype=torch.uint8)
        for i, mp in enumerate(masks_path):
            masks[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))
 
        #mask_visib
        masks_visib_path = list(sorted(glob.glob(os.path.join(self.root, self.split, scene_id, "mask_visib", f"{im_id}_*.png"))))
        masks_visib = torch.zeros((len(masks_path), img.size[1], img.size[0]), dtype=torch.uint8)
        print(masks_visib.shape)
        for i, mp in enumerate(masks_visib_path):
            masks_visib[i] = torch.from_numpy(np.array(Image.open(mp).convert("L")))
        print(masks_visib.shape)
        # create a single label image
        label = torch.zeros((img.size[1], img.size[0]), dtype=torch.int64)
        for i,id in enumerate(obj_ids):
            #print(id, np.sum(masks_visib[i].numpy()==255))
            label[masks_visib[i]==255] = id
        print(masks_visib.shape)
 
        # Bounding boxes for each mask
        scene_gt_infos_item = self.scene_gt_infos[int(scene_id)] if self.split == 'train_pbr' else self.scene_gt_infos[int(scene_id)-1]
        with open(scene_gt_infos_item) as f:
            print("f_bbx:",f)
            scene_gt_info = json.load(f)[str(int(im_id))]
        boxes = [gt['bbox_visib'] for gt in scene_gt_info]
        #print(boxes)
 
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels_detection"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["masks"] = masks
        target["masks_visib"] = masks_visib
        target["scene_id"] = torch.tensor([int(scene_id)])
        target["image_id"] = torch.tensor([int(im_id)])
        target["label"] = label # masken Bild
 
        # Data Augmentations
        if self.transforms is not None:
            img = self.transforms(img)
 
        return img, target
    
 
    def __len__(self):
        return len(self.imgs)
 
        
 
if __name__ == '__main__':
    dataset = TLESSDataset(root='./data/tless', transforms=None, split='train_pbr')
    num_imgs = len(dataset)
    img, target = dataset[3002]
    unique_values = set()
    for mask in target['masks']:
        unique_values.update(torch.unique(mask).tolist())
    
    image = TF.to_tensor(img)
    print("iimage:", type(image))
    print("label:", type(target["label"]))
    print("mask:", type(target["masks"]))
    print("mask_visib:", type(target["masks_visib"]))
    print('num_imgs:',num_imgs)
    print('labels: ', unique_values)
    print('contains classes: ', torch.unique(target['labels_detection']).tolist())
    print('test: ', torch.unique(target["label"]).tolist())
    print('image:', image.shape)
    
    label_array = target["label"].numpy()
    img_array = np.array(img)
    print(label_array)
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
    

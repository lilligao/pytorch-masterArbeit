import sys
# setting path
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/src/')
sys.path.append('/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/')
from datasets.tless import TLESSDataset
import numpy as np
from models.segformer import SegFormer
import torch
import time
from itertools import groupby
import json
from torchmetrics.classification import BinaryJaccardIndex
import torchvision.transforms.functional as TF
import torchmetrics
import config
import matplotlib.pyplot as plt
from lib.bop_toolkit.bop_toolkit_lib.pycoco_utils import binary_mask_to_rle
from PIL import Image



if __name__ == '__main__':
    # assert(config.LOAD_CHECKPOINTS!=None)
    # path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    # model = SegFormer.load_from_checkpoint(path)
    #path = "/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/results/model2_example/general/goodscene/dropout50/"
    path = "/home/lilligao/kit/masterArbeit/pytorch-masterArbeit/results/model2_example/compare_dropouts/dropout_default/"
    # id_imgs = ["0022","0035","0373", "0393"] # 
    id_imgs = ["0184","0807"]
    for id_img in id_imgs:
        target_path = path + id_img + "_target_label.png"
        img_tgt = Image.open(target_path).convert('L')
        img_tgt = TF.to_tensor(img_tgt)
        #print(img_tgt)
        print(torch.unique(img_tgt))
        obj_tgt = torch.unique(img_tgt).tolist()

        predicted_path = path + id_img + "_predicted_label.png"
        img_pd = Image.open(predicted_path).convert('L')
        img_pd = TF.to_tensor(img_pd)
        #print(img_pd)
        print(torch.unique(img_pd))
        ojb_pd = torch.unique(img_pd).tolist()
        
        objs = obj_tgt + list(set(ojb_pd) - set(obj_tgt))
        num_class = len(objs)
        print(num_class)
        print(objs)
        

        class_map = dict(zip(objs, range(num_class)))

        for class_i in class_map:
            print(class_i)
            img_tgt[img_tgt == class_i] = class_map[class_i]
            img_pd[img_pd == class_i] = class_map[class_i]

        print(img_tgt)
        print(torch.unique(img_tgt))
        print(img_pd)
        print(torch.unique(img_pd))

        test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_class) #,ignore_index=0
        iou = test_iou(img_pd,img_tgt).item()
        print("iou: ",iou)

        test_iou_ignore = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_class,ignore_index=0) #,ignore_index=0
        iou_ig = test_iou_ignore(img_pd,img_tgt).item()
        print("iou: " + str(iou_ig) + " ignore index")

        txt_file_path = path + id_img + "_info.txt"
        with open(txt_file_path, "a") as myfile:
            myfile.write("iou: "+str(iou)+ "\n")
            myfile.write("iou: " + str(iou_ig) + " ignore index")







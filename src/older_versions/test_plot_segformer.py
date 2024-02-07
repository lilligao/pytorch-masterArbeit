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
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # assert(config.LOAD_CHECKPOINTS!=None)
    # path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    # model = SegFormer.load_from_checkpoint(path)
    model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")

    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    

    images, target = dataset[7]
    labels = target["label"]
    scene_id = target["scene_id"]
    image_id =target["image_id"]

    print("scene: " + str(scene_id) + ", image: " + str(image_id) + " done")
    # print("test image shape",images.shape)
    # print("test label shape",labels.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    labels = labels.to(device)

    images = images.unsqueeze(0)
    model.eval()
    # print("images", images.shape)
    # print("target", labels.shape)
    loss, logits = model(images, labels)

    upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    preds = torch.softmax(upsampled_logits, dim=1)

    prediction_map = torch.argmax(preds, dim=1, keepdim=True)

    img = images.squeeze().permute(1, 2, 0)
    img_array = np.array(img)

    prediction_array = np.array(prediction_map.squeeze())

    fig,ax = plt.subplots()
    #print("img_array.shape",img_array.shape)
    fig.frameon = False
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prediction_array, cmap='gist_ncar', vmin=0, vmax=31) # so for feste Klasse feste Farbe
    fig.savefig('data/tless/predict_img'+str(scene_id)+'_'+str(image_id)+'.png')
    plt.close()


    # if config.TEST_MODE=="MCDropout":
    #         ## Auskommentierte Sachen sind für MC-Dropout, das sollte man dann aber nicht während dem Training durchlaufen lassen, sondern im Anschluss, wenn man einen finalen Checkpoint hat
    #         # Activate dropout layers
    #         for m in model.modules():
    #             if m.__class__.__name__.startswith('Dropout'):
    #                 m.train()

    #         # record time
    #         torch.cuda.synchronize()  # wait for move to complete
    #         start = torch.cuda.Event(enable_timing=True)
    #         end = torch.cuda.Event(enable_timing=True)
    #         start.record()
    #         # For 5 samples
    #         sample_outputs = torch.empty(size=[config.NUM_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[-2], images.shape[-1]], device=device)
    #         for i in range(config.NUM_SAMPLES):
    #             loss, logits = model(images, labels.squeeze(dim=1))
    #             upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

    #             sample_outputs[i] = torch.softmax(upsampled_logits, dim=1)
            
    #         end.record()
    #         torch.cuda.synchronize()  # need to wait once more for op to finish
            
    #         probability_map = torch.mean(sample_outputs, dim=0)
    #         prediction_map = torch.argmax(probability_map, dim=1, keepdim=True) #1*1*540*720

    #         # Compute the predictive uncertainty
    #         standard_deviation_map = torch.std(sample_outputs, dim=0) #1*31*540*720
    #         predictive_uncertainty = torch.zeros(size=[images.shape[0], images.shape[2], images.shape[3]], device=device) # 1*540*720
            
    #         for i in range(config.NUM_CLASSES):
    #             #standard_deviation_map[:, i, :, :] 1*540*720
    #             predictive_uncertainty = torch.where(prediction_map.squeeze(0) == i, standard_deviation_map[:, i, :, :], predictive_uncertainty)

    #         entropy_map = torch.sum(-probability_map * torch.log(probability_map + 1e-6), dim=1, keepdim=True) #1*1*540*720

    #         # binary map output
    #         binary_accuracy_map = (prediction_map == labels.squeeze(dim=1)).float()

            
    







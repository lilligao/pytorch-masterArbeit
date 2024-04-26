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
    path = "/home/lilligao/kit/masterArbeit/results/compare_dropouts/"
    # id_imgs = ["0022","0035","0373", "0393"] # 
    id_imgs = "0807_std_dropout_default" #0807 / 0184 std/entropy

    target_path = path + id_imgs + ".png"
    img_tgt = Image.open(target_path).convert('L')
    img_tgt = TF.to_tensor(img_tgt)
    print(img_tgt)

    threshold = 0.001708479132503271

    binary_uncertainty_maps = img_tgt>threshold
    print(binary_uncertainty_maps.shape)
    binary_uncertainty_maps = binary_uncertainty_maps.squeeze().numpy()

    # plot binary map 
    fig,ax = plt.subplots()
    fig.frameon = False
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.set_size_inches(720/100,540/100)
    fig.add_axes(ax)
    ax.imshow(binary_uncertainty_maps,cmap='gray') # so for feste Klasse feste Farbe
    fig.savefig(path + id_imgs+'_'+ 'binary_map_uncertainty.png', dpi=100)
    plt.close()
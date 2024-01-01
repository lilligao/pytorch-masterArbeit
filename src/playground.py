import lightning as L
from datasets.tless import TLESSDataset
import numpy as np
import matplotlib.pyplot as plt
import config
from models.segformer import SegFormer
from torch.utils.data import DataLoader
import train
import torch

if __name__ == '__main__':
    model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    img, target = dataset[0]
    img = img.unsqueeze(0)
    #loss, logits = self.model(images, labels.squeeze(dim=1))
    
    pred = model(img)
    print("length of tuple:",len(pred)) # gives a tuple back
    pred = pred[0]
    print("shape of prediction", pred.shape)

    pred = torch.nn.functional.interpolate(pred, size=img.shape[-2:], mode="bilinear", align_corners=False)
    print("shape of prediction after interpolate",pred.shape)
    pred = torch.softmax(pred, dim=1) # normalize
    print("shape of prediction after softmax",pred.shape)
    pred = torch.argmax(pred, dim=1) # the maximum element
    print("shape of prediction after argmax",pred.shape)
    pred = pred.squeeze(0) # delete the first dimension
    print("shape of prediction after squeeze",pred.shape)
    # print(pred.numpy())
    # print(type(pred.numpy().astype(np.uint8)))
    # print(pred.numpy().ndim)

    print('contains classes: ', torch.unique(target["label"]).tolist())
    print('contains classes prediction: ', torch.unique(pred).tolist())
    img_array = img.squeeze(0)
    img_array = img_array.permute(1,2,0).numpy()
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(img_array)

    # torch softmax -> wahrscheinlichkeiten & argmax-> ein layer, am Ende zu numpy() um from gpu zu numpy 
    #print(pred.shape)
    ax[1,0].imshow(pred)
    ax[1,1].imshow(target["label"].squeeze(0).numpy())
    plt.savefig('data/tless/label_img_test.png')
    plt.close()
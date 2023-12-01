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
    model = SegFormer.load_from_checkpoint("./checkpoints/Attempt1/epoch=43-step=68772.ckpt")
    model= model.model
    dataset = TLESSDataset(root='./data/tless', split='train_pbr',step="test")
    num_imgs = len(dataset)
    print(num_imgs)
    img, target = dataset[1]
    img = img.unsqueeze(0)
    #loss, logits = self.model(images, labels.squeeze(dim=1))
    
    # ???? gives only 30 classes back, what about the background class? use the target label to delete the background?
    pred = model(img)
    print(len(pred)) # gives a tuple back
    pred = model(img)[0]
    print(pred.shape)

    pred = torch.nn.functional.interpolate(pred, size=img.shape[-2:], mode="bilinear", align_corners=False)
    print(pred.shape)
    pred = torch.softmax(pred, dim=1) # normalize
    print(pred.shape)
    pred = torch.argmax(pred, dim=1) # the maximum element
    print(pred.shape)
    pred = pred.squeeze(0) # delete the first dimension
    print(pred.shape)
    print(pred.numpy())
    print(type(pred.numpy().astype(np.uint8)))
    print(pred.numpy().ndim)

    label_array = target.numpy()
    print(target.shape)
    print(target.squeeze(0).shape)
    img_array = img.squeeze(0)
    img_array = img_array.permute(1,2,0).numpy()
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(img_array)

    # torch softmax -> wahrscheinlichkeiten & argmax-> ein layer, am Ende zu numpy() um from gpu zu numpy 
    #print(pred.shape)
    ax[1].imshow(pred)
    plt.savefig('data/tless/label_img_test.png')
    plt.close()
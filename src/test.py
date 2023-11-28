import lightning as L
from datasets.tless import TLESSDataset
import numpy as np
import matplotlib.pyplot as plt
import config
from models.segformer import SegFormer
from torch.utils.data import DataLoader
import train

if __name__ == '__main__':
    model = SegFormer.load_from_checkpoint("./checkpoints/Attempt1/epoch=43-step=68772.ckpt")
    model= model.model
    dataset = TLESSDataset(root='./data/tless', split='test_primesense')
    num_imgs = len(dataset)
    img, target = dataset[5]
    img = img.unsqueeze(0)
    predict = model(img)[0] 

    label_array = target.numpy()
    img_array = img.squeeze(0)
    img_array = img_array.permute(1,2,0).numpy()
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(img_array)
    # torch softmax -> wahrscheinlichkeiten & argmax-> ein layer, am Ende zu numpy() um from gpu zu numpy 
    print(predict.shape)
    ax[1].imshow(predict)
    plt.savefig('data/tless/label_img_test.png')
    plt.close()

    # initialize the Trainer
    #trainer = Trainer()  # trainer load checkpoint?? geht auch nicht

    #test the model
    #trainer.test(model, dataloaders=DataLoader(dataset)) # which dataset am besten for testing？？？ which criterion for test dataset???
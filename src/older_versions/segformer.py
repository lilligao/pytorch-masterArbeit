# -*- coding: utf-8 -*-
import torch
from torch import nn
from  torch.optim import lr_scheduler
from datareader import *
from torch.utils.tensorboard import SummaryWriter
from mmseg.models import MixVisionTransformer
from mmseg.models.losses import CrossEntropyLoss

from torch.utils.data import DataLoader
import time
from segformer_pytorch import Segformer

# define the device for training
device = torch.device("cpu")

transform_compose = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
#transform_compose =transforms.ToTensor()
train_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='train_pbr')
test_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='test_primesense')

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training dataset is:{}".format(train_data_size))
print("Length of testing dataset is:{}".format(test_data_size))

# use DataLoader to load data
train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

model = Segformer(
    dims = (64, 128, 256, 512),      # dimensions of each stage
    heads = (1, 2, 4, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 512,              # decoder dimension
    num_classes = 31                 # number of segmentation classes
)

# x = torch.randn(1, 3, 512, 512)
# pred = model(x) # (1, 4, 128, 128)  # output is (H/4, W/4) map of the number of segmentation classes
# print(pred.shape)


# loss function
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# optimizer
learning_rate = 0.01
#learning_rate = 6e-5
optimizer = torch.optim.AdamW(model.parameters(),lr=0.00006,betas=(0.9, 0.999),weight_decay=0.01)
scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=1e-6,total_iters=15)  #total_iters = 1500
scheduler2 = lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=150)  #total_iters = 150000
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# set parameters for training and testing
total_train_step = 0
total_test_step = 0
epoch = 1

# tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------Training Epoche Nr. {} -------".format(i+1))

    # start training
    model.train()
    for data in train_dataloader:
    #     print(1)
        img, target = data
        img = img.to(device)
        #print("original img size:", img.shape)

        # resize target image
        transform_resize = transforms.Resize((512,512))
        target_seg = transform_resize(target["label"])
        #print("target size: {}", targets["label"].shape)
        #print("target size after transform:",target_seg.shape)

        #targets["label"] = targets["label"].to(device)
        prediction = model(img)
        #print("predicted size:",prediction.shape)

        #writer.add_image("output", output,step, dataformats='CHW')
        # double size the prediction
        prediction = nn.functional.interpolate(prediction, size=[512, 512], mode="nearest")
        #print("predicted size after interpolate:",prediction.shape)

        # # extracting maximum values for the pixel
        # prediction = torch.argmax(prediction, 1)
        # print("predicted size after argmax: {}",prediction.shape)

        loss = loss_fn(prediction, target_seg)

        # optimizing the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 1 == 0:
            end_time = time.time()
            print("Training time:",end_time-start_time)
            print("Training steps: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        print('---------')


    # # start testing
    # model.eval()
    # total_test_loss = 0
    # total_accuracy = 0
    # with torch.no_grad():
    #     for data in test_dataloader:
    #         imgs, targets = data
    #         imgs = imgs.to(device)
    #         targets = targets.to(device)
    #         outputs = model(imgs)
    #         loss = loss_fn(outputs, targets)
    #         total_test_loss = total_test_loss + loss.item()
    #         accuracy = (outputs.argmax(1) == targets).sum()
    #         total_accuracy = total_accuracy + accuracy

    # print("整体测试集上的Loss: {}".format(total_test_loss))
    # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    # total_test_step = total_test_step + 1

    # torch.save(model, "tudui_{}.pth".format(i))
    # print("模型已保存")

writer.close()

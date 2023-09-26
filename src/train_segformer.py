# -*- coding: utf-8 -*-
import torch
import torchvision
from datareader import *
from torch.utils.tensorboard import SummaryWriter
from mmseg.models import MixVisionTransformer
from mmseg.models.losses import CrossEntropyLoss

from torch import nn
from torch.utils.data import DataLoader
import time
# define the device for training
device = torch.device("cpu")

train_data = TLESSDataset(root='./data/tless', transforms=torchvision.transforms.ToTensor(), split='train_pbr')
test_data = TLESSDataset(root='./data/tless', transforms=torchvision.transforms.ToTensor(), split='test_primesense')

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training dataset is:{}".format(train_data_size))
print("Length of testing dataset is:{}".format(test_data_size))

# use DataLoader to load data
train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

model = MixVisionTransformer(in_channels=3,out_indices=[3],strides=[4,2,2,2])
model = model.to(device)

# loss function
loss_fn = CrossEntropyLoss()
# loss_fn = loss_fn.to(device)
# optimizer
# learning_rate = 0.01
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
        imgs, targets = data
        #imgs = imgs.to(device)
        outputs = model(imgs)
        #print(targets["label"])
        step=0
        for output in outputs:
            step+=1
            print(output.shape)
            #writer.add_image("output", output,step, dataformats='CHW')
        print('---------')
        print(targets["label"].shape)
        print('---------')
        # loss = loss_fn(outputs, targets["label"])

        # # optimizing the model
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # total_train_step = total_train_step + 1
        # if total_train_step % 100 == 0:
        #     end_time = time.time()
        #     print(end_time-start_time)
        #     print("Training steps: {}, Loss: {}".format(total_train_step, loss.item()))
        #     writer.add_scalar("train_loss", loss.item(), total_train_step)

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

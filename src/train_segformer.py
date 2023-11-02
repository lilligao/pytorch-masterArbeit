# -*- coding: utf-8 -*-
import torch
from torch import nn
from  torch.optim import lr_scheduler
from datareader import *
from torch.utils.tensorboard import SummaryWriter
from mmseg.models import MixVisionTransformer, SegformerHead
from  model_segFormer import *
from torch.utils.data import DataLoader
import time
# define the device for training
device = torch.device("cpu")

transform_compose = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
#transform_compose =transforms.ToTensor()
# random cropping
train_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='train_pbr')
# for test no resizing
test_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='test_primesense')

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training dataset is:{}".format(train_data_size))
print("Length of testing dataset is:{}".format(test_data_size))

# use DataLoader to load data
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)  #batchsize k learning rate wurzel k
# ??batch size can't be bigger
#out_indices=[3]
model_endocder = MixVisionTransformer(in_channels=3,strides=[4, 2, 2, 2],drop_rate=0.1)
#model_endocder = model_endocder.to(device)
model_decoder = SegformerHead(in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=30, # !!! 30
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
#model_dedocder = model_endocder.to(device)
model = SegFormerModel(model_endocder, model_decoder)

# loss function
loss_fn = nn.CrossEntropyLoss()
# !!! Ignore index
#loss_fn = loss_fn.to(device)
# optimizer
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#learning_rate = 6e-5
optimizer = torch.optim.AdamW(model.parameters(),lr=0.00002,betas=(0.9, 0.999),weight_decay=0.01)
scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=1e-6,total_iters=15)  #total_iters = 1500
scheduler2 = lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=150)  #total_iters = 150000
#nur poly LR


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
        target_seg = transform_resize(target)
        #print("target size: {}", targets["label"].shape)
        print("target size after transform:",target_seg.shape)

        #targets["label"] = targets["label"].to(device)
        prediction = model(img)
        print("predicted size:",prediction.shape)
        print("predicted size:",prediction.shape[-2:])

        #writer.add_image("output", output,step, dataformats='CHW')
        # double size the prediction
        prediction = nn.functional.interpolate(prediction, size=[512, 512], mode="nearest") ##bilinear, align_corner
        print("predicted size after interpolate:",prediction.shape)

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

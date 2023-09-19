import torchvision
from torchvision import transforms
from torch import nn
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

vit = torchvision.models.vit_b_16(weights=2, progress=True)
print(vit)
# Transforms ToTensor
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(image)
print(img_tensor.shape)
img_reshape = torch.reshape(img_tensor, (1, 3, 480, 640))
print(img_tensor.shape)
trans_crop = transforms.CenterCrop(224)
img_crop = trans_crop(img_reshape)
outputs = vit(img_crop)
print(outputs)

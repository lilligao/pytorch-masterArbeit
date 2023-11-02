from datareader import *
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from torch.utils.data import DataLoader
transform_compose = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
#transform_compose =transforms.ToTensor()
train_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='train_primesense')
test_data = TLESSDataset(root='./data/tless', transforms=transform_compose, split='test_primesense')

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training dataset is:{}".format(train_data_size))
print("Length of testing dataset is:{}".format(test_data_size))

id2label = {k: str(k) for k in range(31)}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 


def train_transforms(example_batch):
    images = [jitter(imgs) for imgs,targets in example_batch]
    labels = [targets["label"] for imgs,targets in example_batch]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [imgs for imgs,targets in example_batch]
    labels = [targets["label"] for imgs,targets in example_batch]
    inputs = processor(images, labels)
    return inputs


# Set transforms
train_ds = train_transforms(train_data)
test_ds = val_transforms(test_data)


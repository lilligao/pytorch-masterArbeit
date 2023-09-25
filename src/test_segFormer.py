from mmseg.models import MixVisionTransformer
import torch

self = MixVisionTransformer(in_channels=1)
self.eval()
inputs = torch.rand(1, 1, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
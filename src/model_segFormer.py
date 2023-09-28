# Define the combined model
import torch
import torch.nn as nn

class SegFormerModel(nn.Module):
  def __init__(self, encoder, decoder):
    super(SegFormerModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    
  def forward(self, x):
    # Apply the MLP to the feature vectors
    x = self.encoder(x)
    # Apply the GNN to the graph and feature vectors
    x = self.decoder(x)
    return x
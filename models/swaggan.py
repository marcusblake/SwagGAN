import torch
import torch.nn as nn



class SwagGAN(nn.Module):
    def __init__(self):
      super().__init__()
      self.segementation = None
      self.pose = None
      self.clip = None
      self.encoder = None
      self.generator = None

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
      pass
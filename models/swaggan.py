import torch
import torch.nn as nn

from .pose import DenseNet
from .clip import CLIPModel
from .segmentation import SegModel




class SwagGAN(nn.Module):
    def __init__(self):
      super().__init__()
      self.segementation = SegModel()
      self.pose = DenseNet()
      self.clip = CLIPModel()
      self.encoder = None
      self.generator = None

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
      pass
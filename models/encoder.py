from fice_encoder.model import E4e
import torch.nn as nn
import torch
from PIL import Image
from typing import List


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.e4e = E4e(pretrained=pretrained)

    def forward(image: List[Image.Image]) -> torch.Tensor:
        pass
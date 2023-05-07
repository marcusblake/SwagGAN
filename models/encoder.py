from fice_encoder.model import load_e4e
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, pretrained_model_path: str):
        super().__init__()
        self.e4e = load_e4e(pretrained_model_path)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.e4e(image)
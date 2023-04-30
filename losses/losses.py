import torch
import torch. nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos(image_embedding, text_embedding)
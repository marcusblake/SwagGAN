import torch
import torch.nn as nn
from fice_stylegan.model import StyleGAN


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan = StyleGAN()
    
    def forward(latent_rep: torch.Tensor) -> torch.Tensor:
        pass

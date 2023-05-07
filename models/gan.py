import torch
import torch.nn as nn
from fice_stylegan.model import load_gan


class GAN(nn.Module):
    def __init__(self, pretrained_model_path: str):
        super().__init__()
        self.gan = load_gan(pretrained_model_path)
    
    def forward(self, latent_rep: torch.Tensor) -> torch.Tensor:
        """
        Takes in latent representation of an image and returns generates an image.
        Latent representation should be 256 x 256 and the output image will be 256 x 256.
        """
        return self.gan(latent_rep)

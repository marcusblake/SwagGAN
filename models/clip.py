import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel


DEFAULT_CLIP_MODEL = 'openai/clip-vit-base-patch32"'


class CLIPModel(nn.Module):
    def __init__(self, text_encoder: nn.Module = None, image_encoder: nn.Module = None):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        if text_encoder == None:
            self.text_encoder = CLIPTextModel.from_pretrained(DEFAULT_CLIP_MODEL)
        if image_encoder == None:
            self.image_encoder = CLIPVisionModel.from_pretrained(DEFAULT_CLIP_MODEL)


    def forward(self, text_batch: torch.Tensor, image_batch: torch.Tensor) -> torch.Tensor:
        pass
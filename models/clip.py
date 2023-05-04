import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer
from typing import List, Tuple
from PIL import Image


DEFAULT_CLIP_MODEL = 'openai/clip-vit-base-patch32"'
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"

class CLIPModel(nn.Module):
    def __init__(self, dropout: float = 0):
        super().__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(FASHION_CLIP_MODEL)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(FASHION_CLIP_MODEL)
        self.image_processor = CLIPImageProcessor.from_pretrained(FASHION_CLIP_MODEL)
        self.tokenizer = Tokenizer()
        self.dropout = nn.Dropout(dropout)
        self.transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def encode_images(self, image_batch: List[Image.Image]) -> torch.Tensor:
        input_ = self.image_processor(images=image_batch, padding=True, return_tensors='pt').pixel_values
        input_ = self.transform(input_)
        input_ = self.dropout(input_)
        return self.image_encoder(input_).image_embeds
    
    def encode_text(self, text_batch: List[str]):
        input_ = self.tokenizer.tokenize(text_batch)
        return self.text_encoder(**input_).text_embeds

    def forward(self, text_batch: List[str], image_batch: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run text and images through each model
        text_embeds = self.encode_text(text_batch)
        image_embeds = self.encode_images(image_batch)
        return text_embeds, image_embeds

class Tokenizer:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(FASHION_CLIP_MODEL)

    def tokenize(self, text: List[str]) -> torch.Tensor:
        return self.tokenizer(text=text, return_tensors='pt', padding=True)
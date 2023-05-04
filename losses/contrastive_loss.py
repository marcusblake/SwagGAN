import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        The goal of this loss function is to push embeddings that should be similar closer together,
        and to push embeddings that should be far apart further away from each other.
        """
        # Normalize embeddings

        text_embeds, image_embeds = F.normalize(text_embeds, dim=1), F.normalize(image_embeds)
        # Need to calculate dot products 
        logits = (text_embeds @ image_embeds.T) # N x N matrix.
        batch_size = text_embeds.shape[0]
        labels = torch.arange(batch_size, device=text_embeds.device)

        # Perform cross entropy loss
        return 0.5 * F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
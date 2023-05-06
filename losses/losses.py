import torch
import torch. nn as nn

def img_loss_fn(im1, im2):
    loss = F.mse_loss(im1, im2, reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def shape_loss_fn(im1, im2):
    loss = F.mse_loss(im1, im2, reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def clip_loss_fn(clip_similarity):
    loss = (1 - clip_similarity).sum()
    return loss

def w_delta_loss_fn(w):
    N = w.size(1)
    w_ref = w[:, 0].unsqueeze(1).repeat(1, N-1, 1)
    w_tar = w[:, 1:]
    loss = F.mse_loss(w_ref, w_tar, reduction='none')
    loss = loss.mean((1,2)).sum()
    return loss

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos(image_embedding, text_embedding)
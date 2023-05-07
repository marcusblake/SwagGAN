import torch
import torch.nn.functional as F

def img_loss_fn(im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(im1, im2, reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def shape_loss_fn(im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(im1.float(), im2.float(), reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def clip_loss_fn(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    image_embeddings, text_embeddings = F.normalize(image_embeddings, dim=1), F.normalize(text_embeddings, dim=1)
    cosine_similarity = text_embeddings @ image_embeddings.T
    loss = (1 - cosine_similarity).sum()
    return loss

def w_delta_loss_fn(w: torch.Tensor) -> torch.Tensor:
    """
    Corresponds to latent code regularization in the original FICE paper.
    """
    N = w.size(1)
    w_ref = w[:, 0].unsqueeze(1).repeat(1, N-1, 1)
    w_tar = w[:, 1:]
    loss = F.mse_loss(w_ref, w_tar, reduction='none')
    loss = loss.mean((1,2)).sum()
    return loss
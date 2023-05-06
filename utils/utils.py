import torch
from PIL import Image

SEGM_HEAD_LABEL = 1

def image_stich(original_image: torch.Tensor, generated_image: torch.Tensor, segm: torch.Tensor) -> torch.Tensor:  
    segm_head_mask = (segm==SEGM_HEAD_LABEL).int()
    segm_body_mask = 1 - segm_head_mask
    head = torch.mul(original_image, segm_head_mask)
    body_and_background = torch.mul(generated_image, segm_body_mask)
    return head + body_and_background


def expand2square(pil_img: Image.Image, background_color: int):
    """
    Takes a PIL image and resizes it to be a square image while adding
    zero padding of the specified color.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
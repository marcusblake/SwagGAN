import torch
from PIL import Image

SEGM_HEAD_LABEL = 1
SEGM_BODY_LABEL = 2
BLACK_BACKGROUND_COLOR = (0,0,0)

def image_stitch(original_image: torch.Tensor, generated_image: torch.Tensor, segm: torch.Tensor) -> torch.Tensor: 
    """
    Stitching the original image with the predicted image.
    
    Parameters:
        original_image - The original image that was used to create the generated image.
        generated_image - The generated image from StyleGAN
        segm - The head segmentation instance.

    Returns:
        The head from the original image stitched onto the body of the generated image.
    """ 
    segm_head_mask = (segm==SEGM_HEAD_LABEL).int()
    segm_body_mask = 1 - segm_head_mask
    head = torch.mul(original_image, segm_head_mask)
    body_and_background = torch.mul(generated_image, segm_body_mask)
    return head + body_and_background


def get_body_mask(segm: torch.Tensor) -> torch.Tensor:
    return (segm == SEGM_BODY_LABEL).int()

def get_head_mask(segm: torch.Tensor) -> torch.Tensor:
    return (segm == SEGM_HEAD_LABEL).int()

class ConvertToSquareImg(object):
    def __call__(self, pil_img: Image.Image) -> Image.Image:
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), BLACK_BACKGROUND_COLOR)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), BLACK_BACKGROUND_COLOR)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
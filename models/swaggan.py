import torch
import torch.nn as nn

from .pose import DenseNet
from .clip import CLIPModel
from .segmentation import SegModel
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from typing import List
from PIL import Image
from enum import Enum


class ResultDictKeys(Enum):
    GEN_IMAGES = 'gen_images'
    DENSE_POSE_BODY = 'dense_body'
    SEGM_HEAD = 'segm_head'
    TXT_EMBEDDINGS = 'txt_embeds'
    IMG_EMBEDDINGS = 'img_embeds'


def set_instances(box, bs):
    box = Boxes(box)
    instances = [{'pred_classes': torch.tensor([0], device=box.device), 'pred_boxes': box} for _ in range(bs)]
    instances = [Instances(image_size=(256, 256), **x) for x in instances]
    return instances

class SwagGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.generator = None# load_gan()
        self.clip = CLIPModel()
        self.pose = DenseNet()
        self.segmentation = SegModel()
        self.encoder = None
    
    @torch.no_grad()
    def densenet_forward(self, imgs):
        bs = imgs.size(0)
        box = torch.tensor([[0, 0, 256, 256]], device=imgs.device)
        instances = set_instances(box, bs)
        body = self.pose.forward(imgs, instances)
        return body
    
    @torch.no_grad()
    def deeplab_seg_head(self, imgs):
        _, _, head = self.segmentation.forward(imgs)
        return head

    def forward(self, images: torch.Tensor, latent_rep: torch.Tensor, text: List[str]) -> torch.Tensor:
        """
        This forward function takes in a batch of images and list of text descriptions/captions.
        
        Parameters:
            images - Images represented as a PyTorch tensor
            latent_rep - Latent representation of images over which we are optimizing.
        """
        results = {}
        with torch.no_grad():
            generated_imgs = self.generator(latent_rep)
            head = self.deeplab_seg_head(generated_imgs)
            body = self.densenet_forward(generated_imgs)
            text_embeds, img_embeds = self.clip(text, images)
        results['generated_imgs'] = generated_imgs
        results['dense_body'] = body
        results['segm_head'] = head
        results['txt_embeds'] = text_embeds
        results['img_embeds'] = img_embeds
        return results
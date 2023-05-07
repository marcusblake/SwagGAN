import torch
import torch.nn as nn

from .pose import DenseNet
from .clip import CLIPModel
from .segmentation import SegModel
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from typing import List
from PIL import Image

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
     
    def densenet_forward(self, imgs):
        bs = imgs.size(0)
        box = torch.tensor([[0, 0, 256, 256]], device=imgs.device)
        instances = set_instances(box, bs)
        body = self.pose.forward(imgs, instances)
        return body

    def deeplab_seg_head(self, imgs):
        _, _, head = self.segmentation.forward(imgs)
        return head

    def forward(self, images: torch.Tensor, text: List[str]) -> torch.Tensor:
        """
        This forward function takes in a batch of images and list of text descriptions/captions.
        """
        results = {}

        latent_rep = self.encoder(images)
        generated_imgs = self.generator(latent_rep)
        body = self.deeplab_seg_head(generated_imgs)
        head = self.densenet_forward(generated_imgs)
        text_embeds, img_embeds = self.clip(text, images)

        results['latent_rep'] = latent_rep
        results['generated_imgs'] = generated_imgs
        results['dense_body'] = body
        results['segm_head'] = head
        results['txt_embeds'] = text_embeds
        results['img_embeds'] = img_embeds
        
        return results
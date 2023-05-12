import torch
import torch.nn as nn

from .pose import DenseNet
from .clip import CLIPModel
from .segmentation import SegModel
from .gan import GAN
from .encoder import Encoder
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from typing import List
from PIL import Image
from enum import Enum
import torchvision.transforms as T


class ResultDictKeys(Enum):
    GEN_IMAGES = 'gen_images'
    DENSE_POSE_BODY = 'dense_body'
    SEGM_HEAD = 'segm_head'
    TXT_EMBEDDINGS = 'txt_embeds'
    IMG_EMBEDDINGS = 'img_embeds'

class ModelConfig:
    def __init__(self,
                 encoder_pre_trained_path: str,
                 generator_pre_trained_path: str,
                 segmentation_pre_trained_path: str,
                 detectron_config_path: str,
                 clip_pre_trained_path: str):
        self.encoder_pre_trained_path = encoder_pre_trained_path
        self.generator_pre_trained_path = generator_pre_trained_path
        self.segmentation_pre_trained_path = segmentation_pre_trained_path
        self.detectron_config_path = detectron_config_path
        self.clip_path = clip_pre_trained_path

def set_instances(box, bs):
    box = Boxes(box)
    instances = [{'pred_classes': torch.tensor([0], device=box.device), 'pred_boxes': box} for _ in range(bs)]
    instances = [Instances(image_size=(256, 256), **x) for x in instances]
    return instances

class SwagGAN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        
        self.generator = GAN(config.generator_pre_trained_path)
        self.clip = CLIPModel()
        self.clip.load_state_dict(torch.load(config.clip_path))
        self.pose = DenseNet(config.detectron_config_path)
        self.segmentation = SegModel(config.segmentation_pre_trained_path)
        self.encoder = Encoder(config.encoder_pre_trained_path)
        self.to_pil = T.ToPILImage()
    
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

    def forward(self, latent_rep: torch.Tensor, text: str) -> torch.Tensor:
        """
        This forward function takes in a batch of images and an individual text desciption/caption.
        
        Parameters:
            images - Images represented as a PyTorch tensor
            latent_rep - Latent representation of images over which we are optimizing.
        """
        results = {}
        with torch.no_grad():
            generated_imgs = self.generator(latent_rep).clip(0,1)
            head = self.deeplab_seg_head(generated_imgs)
            body = self.densenet_forward(generated_imgs)
            
            n = generated_imgs.shape[0]
            pil_imgs = []
            for i in range(n):
                pil_imgs.append(self.to_pil(generated_imgs[i].cpu().detach()))
            text_embeds, img_embeds = self.clip([text], pil_imgs)
        results[ResultDictKeys.GEN_IMAGES] = generated_imgs.float()
        results[ResultDictKeys.DENSE_POSE_BODY] = body.float()
        results[ResultDictKeys.SEGM_HEAD] = head.float()
        results[ResultDictKeys.TXT_EMBEDDINGS] = text_embeds.float()
        results[ResultDictKeys.IMG_EMBEDDINGS] = img_embeds.float()
        return results
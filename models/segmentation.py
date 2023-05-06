import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from typing import Tuple

PRE_TRAINED_MODEL_PATH = './models/pretrained_models/checkpoint_0040_DeepLabV3_Fashion_Men.pth'

class SegModel(nn.Module): 
    def __init__(self, pretrained_model_path: str = PRE_TRAINED_MODEL_PATH, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = 3
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True) 
        self.model.aux_classifier = None
        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
        self.model.load_state_dict(torch.load(pretrained_model_path))
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.model(x)['out']
        x = F.softmax(x, dim=1)
        
        background = x[:,0].unsqueeze(1)
        head = x[:,1].unsqueeze(1)
        body = x[:,2].unsqueeze(1)
        return background, body, head

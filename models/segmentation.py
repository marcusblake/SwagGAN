import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F

class SegModel(nn.Module):  # todo: move to models
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 3
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True) 
        self.model.aux_classifier = None
        #for param in self.model.parameters():
        #    param.requires_grad = False
        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
        
        self.model.load_state_dict(torch.load('./models/segmentation_model/checkpoint_0040_DeepLabV3_Fashion_Men.pth'))
        self.model.eval()

    def forward(self, x):
        x = self.model(x)['out']
        x = F.softmax(x, dim=1)
        
        background = x[:,0].unsqueeze(1)
        head = x[:,1].unsqueeze(1)
        body = x[:,2].unsqueeze(1)
        return background, body, head

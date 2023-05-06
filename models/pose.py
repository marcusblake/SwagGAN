import os

import torch
import torch.nn as nn

from detectron2.config import CfgNode, get_cfg
from detectron2.engine.defaults import DefaultPredictor  

from densepose import add_densepose_config
from typing import List
from PIL import Image


DETECTRON_CONFIG_PATH = './models/model_configs/densepose_rcnn_R_50_FPN_s1x.yaml'
DETECTRON_MODEL_URL = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

def setup_config(cfg_path, model_path, opts):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_path 
    cfg.INPUT.MIN_SIZE_TEST = 1101
    cfg.INPUT.MAX_SIZE_TEST = 1101
    cfg.freeze()
    return cfg

class DenseNet(nn.Module):
    def __init__(self, indexes=None):
        super().__init__()
        cfg = setup_config(cfg_path = DETECTRON_CONFIG_PATH,
                           model_path = DETECTRON_MODEL_URL,
                           opts = [])
        self.model = DefaultPredictor(cfg).model
        self.body_indexes = indexes or [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor: 
        """
        Image pixel values must be in the range [0,1].

        Normalize images so that pixel values are in between 0 and 1.
        Resize image so that it's square.

        Parameters:
            imgs - Batch of input images to the DensePose model.

        Output:
            body - H X W X 24
        """
        img = img[:, [2, 1, 0], :, :]   # rgb to bgr
        img = img * 255
        height, width = img.shape[-2:]
        dev = img.device

        inputs = [{'image': i_img, 'height': height , 'width': width} for i_img in img]

        output = self.model.inference(inputs, instances, do_postprocess=False)
        output = [x._fields for x in output]

        bbox = [x['pred_boxes'].tensor for x in output]
        poses = [x['pred_densepose'] for x in output] 

        mask_bbox = torch.tensor([x.size(0) > 0 for x in bbox]).to(dev)
        bbox = torch.stack([x[0] if x.size(0) > 0 else torch.ones(4, device=dev) for x in bbox])

        coarse = [x.coarse_segm for x in poses]
        mask_seg = torch.tensor([len(x) > 0 for x in coarse]).to(dev)
        coarse = [x[0] if len(x) > 0 else torch.empty(2, 112, 112, device=dev).fill_(-10000.) for x in coarse]        
        coarse = torch.stack(coarse)
        coarse = torch.softmax(coarse, dim=1)
        coarse = coarse[:, 1].unsqueeze(1)

        fine = [x.fine_segm for x in poses]
        fine = [x[0] if x.size(0) > 0 else torch.empty(25, 112, 112, device=img.device).fill_(-10000.) for x in fine]

        fine = torch.stack(fine)
        fine = torch.softmax(fine, dim=1)

        fine = fine * coarse  
        # mask = mask_seg & mask_bbox

        body = fine[:, self.body_indexes]
        return body

import torch
import torch.nn as nn

class SwagGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.generator = None# load_gan()
        self.clip = None# CLIP
        self.pose = DenseNet()
        self.segmentation = SegModel()
        self.encoder = None
     
    def densenet_forward(self, imgs):
        bs = imgs.size(0)
        box = torch.tensor([[0, 0, 1101, 1101]], device=imgs.device)
        instances = set_instances(box, bs)
        body = self.pose.forward(imgs, instances)
        return body

    def deeplab_seg_head(self, imgs):
        _, _, head = self.segmentation.forward(imgs)

        return head

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        #pbar = tqdm(range(self.opt.n_iters+1))

        #for i in pbar:
            #w = self.w

            #imgs_gen = self.model.gan.gen_w(w)
            body_shape = self.densenet_forward(images)
            head_shape = self.deeplab_seg_head(images)

            print(body_shape.shape)

            loss_dict = {}

            
            loss_dict['shape'] = shape_loss_fn(body_shape, body_shape)
            loss_dict['head_shape'] = shape_loss_fn(head_shape, head_shape)

            #loss_dict['img'] = img_loss_fn(self.img_mask*imgs_gen, self.img_mask*self.imgs_batch)
            #loss_dict['shape'] = shape_loss_fn(self.shape_real, body_shape)
            #loss_dict['head_shape'] = shape_loss_fn(self.head_shape_init, head_shape)
            #loss_dict['w_delta'] = w_delta_loss_fn(w)

            #clip_sim = self.model.clip_similarity(imgs_gen, self.text_feats)
            #loss_dict['clip'] =  clip_loss_fn(clip_sim)

            #loss_dict_scaled = {k: loss_dict[k]*self.opt.weights_dict[k] for k in loss_dict}
            #loss = sum(loss_dict_scaled.values())

            #loss.backward()
            #self.optimizer.step()
            #self.optimizer.zero_grad()
            #pbar.set_description(f'{loss.item():.3f}')

            #if n_log is not None and i % n_log == 0:
            #    ##### image composition
            #    with torch.no_grad():
            #        imgs_final = self.blend_mask * imgs_gen + (1-self.blend_mask) * self.imgs_batch

            #    yield imgs_final, imgs_gen, loss_dict_scaled

            print(loss_dict)

            return loss_dict, body_shape, head_shape
        #with torch.no_grad():
        #    imgs_final = self.blend_mask * imgs_gen + (1-self.blend_mask) * self.imgs_batch
        #yield imgs_final, imgs_gen, loss_dict_scaled
'''
class SwagGAN(nn.Module):
    def __init__(self):
      super().__init__()
      self.segementation = None
      self.pose = None
      self.clip = None
      self.encoder = None
      self.generator = None

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
      pass
      
'''
import argparse
from losses.losses import img_loss_fn, shape_loss_fn, clip_loss_fn, w_delta_loss_fn
from models.swaggan import SwagGAN, ResultDictKeys, ModelConfig
from utils.utils import get_body_mask, get_head_mask, ConvertToSquareImg, image_stitch
import torch
from torchvision import transforms
from torchvision.utils import save_image
from os import listdir
from os.path import isfile, join
from typing import List
import json
from PIL import Image

def get_images_in_directory(directory_path: str) -> List[Image.Image]:
    images = []
    for file in listdir(directory_path):
        full_image_path = join(directory_path, file)
        if isfile(full_image_path):
            images.append(Image.open(full_image_path))
    return images


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def create_model_config():
    with open('swaggan_config.json', 'r') as f:
        config = json.loads(f.read())
        print(config)
        return ModelConfig(
            config['encoder_model'],
            config['generator_model'],
            config['segmentation_model'],
            config['detectron_config'],
            config['clip_model']
        )

def train(lr: float,
          img_path: str,
          txt_description: str,
          epochs: int,
          output_file: str,
          lambda_clip: float,
          lambda_pose: float,
          lambda_latent_reg: float,
          lambda_img: float,
          lambda_head: float) -> None:
    device = get_device()
    model = SwagGAN(create_model_config()).to(device).requires_grad_(False)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256))
    ])

    images = get_images_in_directory(img_path)
    n = len(images)
    transformed_images = torch.stack([trans(image) for image in images]).to(device)

    with torch.no_grad():
        # Pass in initial image to get segmentation.
        _, body, head = model.segmentation.forward(transformed_images)
        latent_rep = model.encoder.forward(transformed_images).requires_grad_(True)
        body_shape = model.densenet_forward(transformed_images)
    head_and_background_mask = 1 - get_body_mask(body)
    org_head_mask = get_head_mask(head)
    # Optimizing over the latent representation space, not model parameters.
    optimizer = torch.optim.Adam([latent_rep], lr=lr)
    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        result = model(latent_rep, txt_description)
        pred_head_segm = result[ResultDictKeys.SEGM_HEAD]
        generated_imgs = result[ResultDictKeys.GEN_IMAGES]
        pred_body_shape = result[ResultDictKeys.DENSE_POSE_BODY]
        txt_embed = result[ResultDictKeys.TXT_EMBEDDINGS]
        img_embed = result[ResultDictKeys.IMG_EMBEDDINGS]

        pred_head_mask = get_head_mask(pred_head_segm)
        clip_loss = clip_loss_fn(img_embed, txt_embed)
        img_loss = img_loss_fn(head_and_background_mask * transformed_images, head_and_background_mask * generated_imgs)
        w_loss = w_delta_loss_fn(latent_rep)
        head_loss = shape_loss_fn(org_head_mask, pred_head_mask)
        pose_loss = shape_loss_fn(body_shape, pred_body_shape)

        print('\t clip_loss {:.3f}, img_loss {:.3f}, w_loss {:.3f}, head_loss {:.3f}, pose_loss {:.3f}'.format(clip_loss.item(), img_loss.item(), w_loss.item(), head_loss.item(), pose_loss.item()))
        loss = lambda_clip * clip_loss + \
                lambda_img * img_loss + \
                lambda_latent_reg * w_loss + \
                lambda_head * head_loss + \
                lambda_pose * pose_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('\t train loss {:.3f}'.format(loss.item()))

    # Output final predictions to output directory.
    result = model(latent_rep, txt_description)
    generated_imgs = result[ResultDictKeys.GEN_IMAGES]
    final_images = image_stitch(transformed_images, generated_imgs, head)
    imgs_compare = torch.cat((transformed_images, final_images), dim=0)
    save_image(imgs_compare, output_file, nrows=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_file', type=str, default='output.png')
    parser.add_argument('--lambda_clip', type=float, default=1)
    parser.add_argument('--lambda_pose', type=float, default=1)
    parser.add_argument('--lambda_latent_reg', type=float, default=2)
    parser.add_argument('--lambda_img', type=float, default=30)
    parser.add_argument('--lambda_head', type=float, default=10)
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)

    args = parser.parse_args()
    train(args.lr,
          args.img_path,
          args.description,
          args.epochs,
          args.output_file,
          args.lambda_clip,
          args.lambda_pose,
          args.lambda_latent_reg,
          args.lambda_img,
          args.lambda_head)


if __name__ == '__main__':
    main()
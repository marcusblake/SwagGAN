import argparse
from losses.losses import img_loss_fn, shape_loss_fn, clip_loss_fn, w_delta_loss_fn
from models.swaggan import SwagGAN, ResultDictKeys
from data.deep_fashion import DeepFashionMultimodalImageAndTextDataset
from utils.utils import get_body_mask, get_head_mask
import torch


def train(lr: float,
          dataset_path: str,
          batch_size: int,
          epochs: int,
          output_file: str,
          lambda_clip: float,
          lambda_pose: float,
          lambda_latent_reg: float,
          lambda_img: float,
          lambda_head: float) -> None:
    

    model = SwagGAN().requires_grad_(False)

    with torch.no_grad():
        # Pass in initial image to get segmentation.
        _, body, head = model.segmentation.forward()
        latent_rep = model.encoder.forward()
        body_shape = model.densenet_forward()

    head_and_background_mask = 1 - get_body_mask(body)
    org_head_mask = get_head_mask(head)
    # Optimizing over the latent representation space, not model parameters.
    optimizer = torch.optim.Adam([latent_rep], lr=lr)
    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        cummulative_loss = 0.0

        result = model(img_batch, latent_rep, text)
        pred_head_segm = result[ResultDictKeys.SEGM_HEAD]
        generated_imgs = result[ResultDictKeys.GEN_IMAGES]
        pred_body_shape = result[ResultDictKeys.DENSE_POSE_BODY]

        pred_head_mask = get_head_mask(pred_head_segm)
        loss = lambda_clip * clip_loss_fn() + lambda_img * img_loss_fn(head_and_background_mask * org_images, head_and_background_mask * generated_imgs) + lambda_latent_reg * w_delta_loss_fn() + \
                lambda_head * shape_loss_fn(org_head_mask, pred_head_mask) + \
                lambda_pose * shape_loss_fn(body_shape, pred_body_shape)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cummulative_loss += loss.item()
        avg_train_loss = cummulative_loss / n
        print('\t train loss {:.6f}'.format(avg_train_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=50, required=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_file', type=str, default='output.png')
    parser.add_argument('--lambda_clip', type=float, default=1)
    parser.add_argument('--lambda_pose', type=float, default=1)
    parser.add_argument('--lambda_latent_reg', type=float, default=2)
    parser.add_argument('--lambda_img', type=float, default=30)
    parser.add_argument('--lambda_head', type=float, default=10)

    args = parser.parse_args()
    train(args.lr,
          args.dataset_path,
          args.batch_size,
          args.epochs,
          args.output_file,
          args.lambda_pose,
          args.lambda_latent_reg,
          args.lambda_img,
          args.lambda_head)


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import argparse
import matplotlib.pyplot as plt
from models.clip import CLIPModel
from losses.contrastive_loss import ContrastiveLoss
from data.deep_fashion import DeepFashionMultimodalImageAndTextDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def custom_collate(batch):
    return {
        'txt': [data['txt'] for data in batch],
        'img': [data['img'] for data in batch]
    }

def train(epochs: int,
          optim: str,
          lr: float,
          dataset_folder: str,
          batch_size: int,
          dropout: float,
          checkpoint_path: str,
          checkpoint_frequency: float) -> None:
    device = get_device()

    model = CLIPModel()
    dataset = DeepFashionMultimodalImageAndTextDataset(dataset_folder=dataset_folder, men_only=True)
    loss_fn = ContrastiveLoss()
    n = 10
    indices = np.arange(n)
    split = int(np.floor(0.2 * n))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:2]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate)

    if optim == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    epochs_ = []
    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1,epochs+1):
        print('Epoch {}/{}:'.format(epoch, epochs))
        epochs_.append(epoch)
        train_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            img_embeds, txt_embeds = model(batch['txt'], batch['img'])
            loss = loss_fn(txt_embeds, img_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_indices)
        train_losses.append(avg_train_loss)
        print('\t train  loss {:.6f}'.format(avg_train_loss))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                img_embeds, txt_embeds = model(batch['txt'], batch['img'])
                loss = loss_fn(txt_embeds, img_embeds)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_indices)
        test_losses.append(avg_test_loss)
        print('\t test  loss {:.6f}'.format(avg_test_loss))

        if epoch % checkpoint_frequency == 0:
            print('Checkpointing model...')
            torch.save(model.state_dict(), f'{checkpoint_path}/clip_model_epoch{epoch}.pt')
            print('Done writing checkpoint.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=str, default='adam', required=False)
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    parser.add_argument('--plot_loss', action='store_true', required=False)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=50, required=False)
    parser.add_argument('--model_checkpoint_path', type=str, default='./')
    parser.add_argument('--checkpoint_frequency', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()
    train(args.epochs, args.optim, args.lr, args.dataset_path, args.batch_size, args.dropout, args.model_checkpoint_path, args.checkpoint_frequency)


    


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import argparse
import matplotlib.pyplot as plt


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model: nn.Module,
          loss_fn: nn.Module,
          epochs: int,
          optim: str,
          lr: float,
          train_data: DataLoader,
          test_data: DataLoader,
          plot_loss: bool = False) -> None:
    device = get_device()
    if optim == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, device=device)
    else:
        optimizer = SGD(model.parameters(), lr=lr, device=device)

    train_losses = []
    test_losses = []
    epochs_ = []
    model.to(device)
    for epoch in range(1,epochs+1):
        model.train()
        epochs_.append(epoch)
        train_loss = 0.0
        for i, (X, y) in enumerate(train_data):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            optimizer.step()
            loss.backward()
        train_losses.append(train_loss / len(train_data))

        model.eval()
        test_loss = 0.0
        for i, (X, y) in enumerate(test_data):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred)
            test_loss += loss.item()
        test_losses.append(test_loss / len(test_data))

        if plot_loss:
            line1, = plt.plot(epochs_, train_loss, color='blue')
            line2, = plt.plot(epochs_, test_loss, color='orange')
            plt.pause(0.001)

    if plot_loss:
        plt.legend([line1, line2], ['Training Loss', 'Test Loss'])
        plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--optim')
    parser.add_argument('--lr')
    parser.add_argument('--plot_loss')
    


if __name__ == "__main__":
    main()
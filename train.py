import torch
import torch.optim as optim
from helper import write_log, write_figures
from hdr_loss import HDRLoss
import numpy as np
from dataset import get_loader

from model import Model
from tqdm import tqdm


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0

    for inputs, targets in tqdm(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        # loss
        loss = criterion(inputs, outputs, targets, separate_loss=True)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
    return epoch_loss


def train(root, device, model, epochs, bs, lr):
    print('start training ...........')
    train_loader, val_loader = get_loader(root=root, batch_size=bs, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = HDRLoss(device)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model(device).to(device)
    batch_size = 8
    num_epochs = 200
    learning_rate = 0.01
    root = 'data/train'
    train(root, device, model, num_epochs, batch_size, learning_rate)

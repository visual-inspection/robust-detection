###########################################################################
# Paste the following 3 lines at the beginning of each app to make it work.
from sys import path; from pathlib import Path; apps = Path(__file__);
while apps.parent.name != 'apps': apps = apps.parent;
path.append(str(apps.parent)); path.append(str(apps.parent.parent)); del apps;
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER;
###########################################################################

"""
Author: Sebastian Hönel
"""

import torch
from torch import manual_seed
from torch import cuda, nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchsummary import summary
from rd.data.ConvNeXt_V2 import Dataset_ConvNeXt_V2
from rd.autoencoders.ConvNeXt_V2_DepthAE import DepthAE, InferenceDepthAE
from rd.tools.EarlyStopper import EarlyStopper





class RMLSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        # rmlse
        return torch.sqrt(torch.mean(torch.log(1.0 + 1e-20 + (pred - actual)**2)))



def train_loop(loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        X, y = sample_batched['input'].to(dev), sample_batched['output'].to(dev)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), idx_batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(loader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    num_batches = len(loader)

    test_loss = 0
    with torch.no_grad():
        for idx_batch, sample_batched in enumerate(loader):
            X, y = sample_batched['input'].to(dev), sample_batched['output'].to(dev)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"\n--- Test Loss:\n{test_loss}")
    return test_loss





if __name__ == '__main__':
    manual_seed(0xbeef)
    batch_size = 28

    train_dataset = Dataset_ConvNeXt_V2(
        scaler=None,
        move_channels_first=True,
        file=DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/train-good-CNxV2.npy'))
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)


    test_dataset = Dataset_ConvNeXt_V2(
        scaler=train_dataset.scaler,
        move_channels_first=True,
        file=DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/test-good-CNxV2.npy'))
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)


    dev = 'cuda' if cuda.is_available() else 'cpu'
    model = DepthAE(dev=dev).eval()
    summary(model=model, batch_size=32, input_size=(2816, 16, 16))
    
    learning_rate = 5e-4
    loss_fn = nn.MSELoss() # RMLSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, amsgrad=True)
    early_stopper = EarlyStopper(patience=35, min_delta=1e-2)

    epochs = 20000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        test_loss = test_loop(loader=test_loader, model=model, loss_fn=loss_fn)
        if early_stopper.early_stop(test_loss):
            torch.save(obj=model.state_dict(), f=MODELS_FOLDER.joinpath(f'./{DepthAE.__name__}_ConvNeXt_V2.torch'))
            break

###########################################################################
# Paste the following 3 lines at the beginning of each app to make it work.
from sys import path; from pathlib import Path; apps = Path(__file__);
while apps.parent.name != 'apps': apps = apps.parent;
path.append(str(apps.parent.parent)); del apps;
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER;
###########################################################################

"""
Author: Sebastian Hönel

In this app, we init the BnQn1d from data instead of training it. Then, we can
use it as a rather static feature extractor.
"""

import numpy as np
import torch
import normflows as nf
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, Tensor, tensor, manual_seed, device
from torch.optim import Optimizer
from rd.tryouts.real_nvp import make_model
from rd.tools.EarlyStopper import EarlyStopper


class XOnlyDataset(Dataset):
    def __init__(self, file: str | Path, even_odd = Literal['even', 'odd', 'both']) -> None:
        super().__init__()
        self.even = even_odd == 'even'
        temp = np.load(file=file)
        self.raw = tensor(temp[::2] if self.even else temp[1::2])
    
    def __len__(self) -> int:
        return self.raw.shape[0]
    
    def __getitem__(self, index) -> Tensor:
        if is_tensor(index):
            index = index.tolist()
        
        # Since we want to train an auto-encoder, X and Y are equal.
        return self.raw[index]



def train_loop(loader: DataLoader, model: nf.NormalizingFlow, optimizer: Optimizer, dev: device, batch_size: int):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        optimizer.zero_grad()
        x = sample_batched.to(dev)

        loss = model.forward_kld(x=x)

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss, current = loss.detach().cpu().item(), idx_batch * batch_size + len(x)
        print(f"loss: {np.log(loss):>2f}  [{current:>4d}/{size:>4d}]")


def test_loop(loader: DataLoader, model: nf.NormalizingFlow, dev: device, return_list: bool=False):
    model.eval()

    losses = []
    test_loss = 0
    with torch.no_grad():
        for idx_batch, sample_batched in enumerate(loader):
            x = sample_batched.to(dev)
            losses.append(model.forward_kld(x=x).item())
    
    test_loss = np.mean(losses)
    print(f"\n--- Test Loss:\n{np.log(test_loss)}")
    if return_list:
        return losses
    return test_loss



if __name__ == '__main__':
    mvtec = DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/')
    train_good = mvtec.joinpath('./train-good-CNxV2_DepthAE_BnQn1d.npy')
    test_good = mvtec.joinpath('./test-good-CNxV2_DepthAE_BnQn1d.npy')
    test_defect = mvtec.joinpath('./test-defect-CNxV2_DepthAE_BnQn1d.npy')

    # All data is vertically stacked, where the first (even) index is the
    # batch-normalized data and the second (odd) index is the quantile-normalized data.
    even_odd = 'even'

    manual_seed(0xbeef)
    loader_train = DataLoader(dataset=XOnlyDataset(file=train_good, even_odd=even_odd), batch_size=28, shuffle=True)
    loader_test_good = DataLoader(dataset=XOnlyDataset(file=test_good, even_odd=even_odd))
    loader_test_defect = DataLoader(dataset=XOnlyDataset(file=test_defect, even_odd=even_odd))

    
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(dev=dev, num_features=loader_train.dataset.raw.shape[1], num_intermittent=1024).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    batch_size = 56
    learning_rate = 5e-4
    early_stopper = EarlyStopper(patience=35, min_delta=1e-2)

    epochs = 2000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(loader=loader_train, model=model, optimizer=optimizer, dev=dev, batch_size=batch_size)
        
        test_loss_good = np.log(test_loop(loader=loader_test_good, model=model, dev=dev))
        test_loss_defect = np.log(test_loop(loader=loader_test_defect, model=model, dev=dev))

        print(f'Log KLdiv\t[good]: {test_loss_good:>3f};\t[defect]: {test_loss_defect:>3f}')
        
        if early_stopper.early_stop(test_loss_good):
            losses_good = test_loop(loader=loader_test_good, model=model, dev=dev, return_list=True)
            losses_defect = test_loop(loader=loader_test_defect, model=model, dev=dev, return_list=True)
            # torch.save(obj=model.state_dict(), f=MODELS_FOLDER.joinpath(f'./{DepthAE.__name__}_ConvNeXt_V2.torch'))
            # import pandas as pd
            # temp = pd.DataFrame({
            #     'good': losses_good,
            #     'defect': losses_defect
            # })
            # temp.to_csv(DATA_FOLDER.joinpath('./temp.csv'), index=False)
            break

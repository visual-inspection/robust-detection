from typing import Literal
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from QuantNorm1D import QuantNorm1D_new

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.2)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train)
X_test_scaled = scaler.transform(X=X_test)



from sklearn.ensemble import RandomForestRegressor
import numpy as np

rf = RandomForestRegressor(random_state=0xbeef)
rf.fit(X=X_train_scaled, y=y_train)


y_true, y_pred = y_test, rf.predict(X=X_test_scaled)
rf_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
print(f'RMSE: {rf_rmse}')




class MyDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index) -> tuple[Tensor]:
        return { 'input': self.X[index], 'output': self.y[index] }


batch_size = 160
ds_train = MyDataset(X=X_train, y=y_train)
train_set = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)

ds_test = MyDataset(X=X_test, y=y_test)
test_set = DataLoader(dataset=ds_test, batch_size=batch_size)

torch.autograd.set_detect_anomaly(True)

dev = 'cuda'

class RegModel(torch.nn.Module):
    def __init__(self, batch_size: int, input_shape: tuple[int], batch_type: Literal['ordinary', 'quantile'], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.q1 = QuantNorm1D_new(
            input_shape=(batch_size, *input_shape),
            num_pit_samples=1000,
            take_num_samples_when_full = 1, # batch_size // 10,
            dev=dev,
            normal_backtransform=True,
            trainable_bandwidths=False)
        self.a1 = torch.nn.SiLU(inplace=True)
        self.l1 = torch.nn.Linear(in_features=8, out_features=40, device=dev)
        self.a2 = torch.nn.SiLU(inplace=True)
        self.l2 = torch.nn.Linear(in_features=40, out_features=1, device=dev)

        self.batch_type = batch_type
        self.batch = self.q1 if batch_type == 'quantile' else torch.nn.BatchNorm1d(num_features=8)

        self.model = torch.nn.Sequential(
            self.batch,
            
            self.a1,

            self.l1,

            self.a2,

            torch.nn.Dropout(p=0.3),

            self.l2
        ).to(device=dev)
    

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


loss_fn = torch.nn.MSELoss() # RMLSELoss()

from torch.optim import Optimizer

def train_loop(loader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: Optimizer):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        X, y = sample_batched['input'].to(dev), sample_batched['output'].to(dev)
        pred = model(X).reshape(y.shape)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), idx_batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(loader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, epoch: int):
    model.eval()
    num_batches = len(loader)

    test_loss = 0
    with torch.no_grad():
        for idx_batch, sample_batched in enumerate(loader):
            X, y = sample_batched['input'].to(dev), sample_batched['output'].to(dev)
            pred = model(X).reshape(y.shape)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"\n--- Test Loss (epoch {epoch}):\n{test_loss}")
    import time
    time.sleep(0.95)
    return test_loss


class EarlyStopper:
    """
    Copy-pasted from https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



models = [
    RegModel(batch_type='quantile', batch_size=batch_size, input_shape=(8,)),
    RegModel(batch_type='ordinary', batch_size=batch_size, input_shape=(8,))]

if __name__ == '__main__':
    for model in models:
        print(f'Random forest RMSE was: {rf_rmse}\n')
        input(f'Press key to start training model: {model.batch_type}')
        optimizer = torch.optim.Adam(params=model.parameters())
        early_stopper = EarlyStopper(patience=15, min_delta=5e-3)

        test_loss = 1e30
        best_loss = 1e30
        best_epoch = 0
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(loader=train_set, model=model, loss_fn=loss_fn, optimizer=optimizer)
            test_loss = test_loop(loader=test_set, model=model, loss_fn=loss_fn, epoch=t)
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = t
            if early_stopper.early_stop(test_loss):
                print(f'Stopping Early after {t} epochs.')
                break
        
        print(f'Final loss after {t} epochs: {test_loss}')
        print(f'Best loss in epoch {best_epoch}: {best_loss}')

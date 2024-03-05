import numpy as np
from numpy import ndarray
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor, is_tensor, device
from torch.optim import Optimizer, Adam
from torchsummary import summary


dev = device('cuda')

# 224, 132, 11, 11
data_path = Path(__file__).resolve().parent.joinpath('../Depth_AE.npy')


class MyDataset(Dataset):
    def __init__(self, file: Path) -> None:
        super().__init__()
        self.raw: ndarray = np.load(file=file)
        self.raw = self.raw.reshape((self.raw.shape[0], 132*11*11))
    
    def __len__(self) -> int:
        return self.raw.shape[0]
    
    def __getitem__(self, index) -> ndarray:
        if is_tensor(index):
            index = index.tolist()
        
        return self.raw[index]



import torch
import torchvision as tv
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm



L = 3
K = 16
torch.manual_seed(0)

input_shape = (132, 11, 11)
n_dims = np.prod(input_shape)
channels = 132
hidden_channels = 256
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                     split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]


# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges).to(device=dev)


batch_size = 28

train_ds = MyDataset(file=data_path)
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)



max_iter = 20000
loss_hist = np.array([])
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    try:
        X = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        X = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(X.to(device=dev))
        
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())




















##################

import normflows as nf

batch_size = 28
# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(132*11*11)

# Define list of flows
num_layers = 4
flows = []


param_map = nf.nets.MLP(layers=[132*11*11, 20*11*11, 128], dropout=0.3)
flows.append(nf.flows.AffineCouplingBlock(param_map=param_map))
flows.append(nf.flows.Permute(num_channels=2, mode='swap'))


for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([128, 64, 16], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))


# If the target density is not given
model = nf.NormalizingFlow(base, flows).to(device=dev)
#summary(model=model, batch_size=batch_size, input_size=(132*11*11,))



def train_loop(loader: DataLoader, model: nf.NormalizingFlow, optimizer: Optimizer):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        x = sample_batched.to(dev)
        loss = model.forward_kld(x)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        loss, current = loss.item(), idx_batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    train_ds = MyDataset(file=data_path)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size)
    optimizer = Adam(params=model.parameters())

    for e in range(100):
        train_loop(loader=train_loader, model=model, optimizer=optimizer)

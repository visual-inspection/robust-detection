import torch
import numpy as np
import normflows as nf
from tqdm import tqdm
from torch import hstack
from torch.distributions import Beta, Gumbel, LogNormal, Pareto
from torch import Tensor, device
from torch.nn import Module, Sequential, Linear, SiLU, Dropout, init



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d1, d2, d3, d4 = Beta(1,2), Gumbel(1, 2), LogNormal(-4, 4), Pareto(1.5, 1.2)
num_features = 12
dist_shape = num_features // 4

def sample(n: int) -> Tensor:
    shape = (n, num_features // 4)
    #return hstack((d1.sample(shape), d2.sample(shape), d3.sample(shape), d4.sample(shape)))
    return hstack((d3.sample(shape), d3.sample(shape), d3.sample(shape), d3.sample(shape)))


class MyMLP(Module):
    def __init__(self, num_in: int, num_out: int, dev: device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Linear -> Relu -> Dropout -> Linear -> Relu
        last = Linear(in_features=64, out_features=num_out)
        init.zeros_(last.bias)
        init.zeros_(last.weight)

        self.model = Sequential(
            Linear(in_features=num_in, out_features=64),
            SiLU(),
            Dropout(p=0.5),
            last,
            SiLU()
        ).to(device=dev)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)





base = nf.distributions.base.DiagGaussian(shape=(num_features,))

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    param_map = MyMLP(num_in=num_features // 2, num_out=num_features, dev=dev)
    # param_map = nf.nets.MLP(layers=[6, 64, 64, 12], output_fn='tanh')

    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(num_channels=num_features, mode='shuffle'))

# Construct flow model
model = nf.NormalizingFlow(base, flows).to(device=dev)







# Train model
max_iter = 2000
num_samples = 2 ** 9
show_iter = 500
loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for it in range(max_iter): #tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    x = sample(n=num_samples).to(device=dev)

    # Compute loss
    loss = model.forward_kld(x)

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        print(loss.to('cpu').data.numpy())
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

print(loss_hist)

import torch
from torch import cuda, is_tensor, manual_seed, nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from numpy import load, ndarray, swapaxes
from sklearn.preprocessing import StandardScaler
from typing import Tuple




class Dataset_ConvNeXTV2(Dataset):
    def __init__(self, file: str, move_channels_first: bool, scaler: StandardScaler = None) -> None:
        super().__init__()
        self.raw: ndarray = load(file=file)
        if isinstance(scaler, StandardScaler):
            # Check if it was fit -> If so, apply (otherwise fit, apply, and store)
            if not hasattr(scaler, 'n_features_in_') or scaler.n_features_in_ == 0:
                # Was not fit before
                scaler.fit(X=self.raw)
            
            self.raw = scaler.transform(X=self.raw)
        
        self.scaler = scaler

        self.raw = self.raw.reshape((self.raw.shape[0], 16, 16, 2816))
        if move_channels_first:
            # 16x16x2816 (channels last)  -->>  2816x16x16 (channels first)
            self.raw = swapaxes(swapaxes(self.raw, 3, 2), 2, 1)
    
    def __len__(self) -> int:
        return self.raw.shape[0]
    
    def __getitem__(self, index) -> ndarray:
        if is_tensor(index):
            index = index.tolist()
        
        # Since we want to train an auto-encoder, X and Y are equal.
        temp = self.raw[index]
        return {'input': temp, 'output': temp}



batch_size = 28

train_dataset = Dataset_ConvNeXTV2(
    scaler=None,
    move_channels_first=True,
    file='/mnt/d/lnu/Repos/cs-flow/data/features/mvtec_cable_eq_features-scoring/train-cnxV2-good.npy')

# for i, sample in enumerate(train_dataset):
#     print((i, sample['input'].shape, sample['output'].shape))

manual_seed(0xbeef)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

# for i_batch, sample_batched in enumerate(train_loader):
#     print((i_batch, sample_batched['input'].shape, sample_batched['output'].shape))


test_dataset = Dataset_ConvNeXTV2(
    scaler=train_dataset.scaler,
    move_channels_first=True,
    file='/mnt/d/lnu/Repos/cs-flow/data/features/mvtec_cable_eq_features-scoring/test-cnxV2-good.npy')
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)


device = 'cuda' if cuda.is_available() else 'cpu'



class PrepDepthwiseBatchNorm1d(nn.Module):
    def __init__(self, inverse: bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inverse = inverse
    
    def forward(self, x):
        if self.inverse:
            x = torch.reshape(x, (x.shape[0], 16, 16, 2816))
            x = torch.swapaxes(torch.swapaxes(x, 3, 2), 2, 1)
        else:
            x = torch.swapaxes(torch.swapaxes(x, 1, 2), 2, 3)
            x = torch.reshape(x, (x.shape[0], 256, 2816))
        return x


class HRepeat(nn.Module):
    def __init__(self, num_repeats=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats
    
    def forward(self, x: Tensor) -> Tensor:
        return x.repeat(repeats=self.num_repeats)


class Reshape(nn.Module):
    def __init__(self, shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape = shape
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(shape=(x.shape[0], *self.shape))
        return x


class Split(nn.Module):
    def __init__(self, *args: nn.Module, **kwargs) -> None:
        super().__init__()

        self.modules = list(args)
        if len(self.modules) == 0:
            raise Exception('Need at least one module.')
        
    def forward(self, x: Tensor) -> Tensor:
        temp = torch.hstack(list([m(x) for m in self.modules]))
        return temp


class MinPool2D(nn.Module):
    def __init__(self, kernel_size: int | Tuple[int, ...], stride: int | Tuple[int, ...] | None = None, padding: int | Tuple[int, ...] = 0, dilation: int | Tuple[int, ...] = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.m2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    
    def forward(self, x: Tensor) -> Tensor:
        return -self.m2d(-x)


class Depth_AE(nn.Module):
    """
    This is the first type of AE on the ConvNeXT-V2 features that will work
    depth-wise. The data comes in 16x16x2816 and we will attempt to convolute
    along the first two axes, considering the many features along the last
    within each kernel's slice.
    The bottleneck of this AE should be a 1x1 convolution which we will later
    use for extracting features.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ae = nn.Sequential(
            # Normalize each feature depth-wise along its deep dimension here.
            # The data comes as 2816x16x16, or, in other words, it comes as 256
            # 2816-valued features.
            PrepDepthwiseBatchNorm1d(),
            # N, C, L:
            nn.BatchNorm1d(num_features=256),
            PrepDepthwiseBatchNorm1d(inverse=True),

            # 2816x16x16  -->>  352x13x13
            #
            # Each filter has 4x4x2816+1=45,057 weights+bias and produces 13x13=169 values
            # We have 352 filters resulting in 15,860,064 weights/biases
            # We have 352x13x13=59,488 outputs
            nn.Conv2d(in_channels=2816, out_channels=352, kernel_size=(4,4), stride=(1,1), padding=0, bias=True),

            nn.SiLU(inplace=True),

            #HRepeat(num_repeats=3),

            # 1x13x13 (Computes a max for each depth-wise feature). While this is what we want,
            # it reduces the amount of features perhaps too drastically.
            #nn.MaxPool3d(kernel_size=(352,1,1), stride=(1,1,1))

            # 352x11x11 (Computes max of 352 13x13 patches, i.e., for each depth-slice/channel).
            # While this can work, this is not what we want, as we consider a feature along a
            # depth-axis.
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=0),

            # Here we learn 128 filters, each applied to a 352x1x1 slice. We will get
            # 128x11x11=15,488 features out of this.
            nn.Conv2d(in_channels=352, out_channels=128, kernel_size=(1,1), stride=1, bias=True),


            #####  BOTTLENECK HERE  #####
            #####  BOTTLENECK HERE  #####
            #####  BOTTLENECK HERE  #####
            # Once trained, we shall cut of the AE between this last convolution and the activation!

            nn.SiLU(inplace=True),

            ##### De-Convolution starts here  #####

            # Now we have to come back to 2816x16x16 from 128x11x11

            nn.ConvTranspose2d(in_channels=128, out_channels=352, kernel_size=(5,5), stride=1),
            nn.SiLU(inplace=True),

            # nn.Flatten(),
            nn.Dropout(inplace=True, p=1./3.),
            # nn.Linear(in_features=352*15*15, out_features=352*15*15)
            # # nn.LeakyReLU(inplace=True),
            # # Reshape(shape=(352,13,13)),

            nn.ConvTranspose2d(in_channels=352, out_channels=2816, kernel_size=(2,2), stride=1),
            nn.LeakyReLU(inplace=True)

        ).to(device=device)
    
    def forward(self, x):
        return self.ae(x)


model = Depth_AE()

summary(model=model, batch_size=32, input_size=(2816, 16, 16))


class RMLSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        # rmlse
        return torch.sqrt(torch.mean(torch.log(1.0 + 1e-20 + (pred - actual)**2)))


learning_rate = 1e-3
loss_fn = nn.MSELoss() # RMLSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

from torch.optim import Optimizer

def train_loop(loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        X, y = sample_batched['input'].to(device), sample_batched['output'].to(device)
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
            X, y = sample_batched['input'].to(device), sample_batched['output'].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"\n--- Test Loss:\n{test_loss}")
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
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=35, min_delta=1e-2)

epochs = 20000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test_loss = test_loop(loader=test_loader, model=model, loss_fn=loss_fn)
    if early_stopper.early_stop(test_loss):
        torch.save(obj=model.state_dict(), f=f'./{Depth_AE.__name__}.torch')
        break

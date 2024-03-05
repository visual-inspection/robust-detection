from torch import nn, Tensor
from numpy import prod


class InversibleFlatten(nn.Module):
    def __init__(self, input_shape: tuple[int], inverse: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self._inverse = inverse

    @property
    def inverse(self) -> 'InversibleFlatten':
        return InversibleFlatten(input_shape=self.input_shape, inverse=not self._inverse)
    

    def forward(self, x: Tensor) -> Tensor:
        if self._inverse:
            assert len(x.shape) == 2
            # Inflate back to original shape:
            return x.reshape((x.shape[0], *self.input_shape))
        
        return x.reshape((x.shape[0], prod(x.shape[1:])))


import torch

x = torch.randn(size=(32,3,16,16))
f = InversibleFlatten(input_shape=x.shape[1:], inverse=False)
x_flat = f.forward(x)

i = f.inverse
x_back = i.forward(x_flat)

assert x.shape == x_back.shape
"""
File: Reshape.py
Author: Sebastian Hönel
"""

from torch import nn, Tensor

class Reshape(nn.Module):
    def __init__(self, shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape = shape
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(shape=(x.shape[0], *self.shape))
        return x

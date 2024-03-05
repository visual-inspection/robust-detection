"""
File: MinPool2d.py
Author: Sebastian Hönel
"""

from torch import nn, Tensor
from typing import Tuple


class MinPool2d(nn.Module):
    def __init__(self, kernel_size: int | Tuple[int, ...], stride: int | Tuple[int, ...] | None = None, padding: int | Tuple[int, ...] = 0, dilation: int | Tuple[int, ...] = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.m2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    
    def forward(self, x: Tensor) -> Tensor:
        return -self.m2d(-x)

"""
File: Split.py
Author: Sebastian Hönel
"""

from typing import Literal, Self
from torch import nn, Tensor
import torch

class Split(nn.Module):
    """
    Makes n copies of the input and gives one to each of the nested sub modules.
    During forward, each module's forward() is called. The results are than stacked
    horizontally or vertically (and, therefore, are  expected to be compatible).
    """
    def __init__(self, stack: Literal['h', 'v'], *args: nn.Module, **kwargs) -> None:
        super().__init__()

        self.stack = stack
        self.modules: list[nn.Module] = list(args)
        if len(self.modules) == 0:
            raise Exception('Need at least one module.')
        
    def forward(self, x: Tensor) -> Tensor:
        func = torch.hstack if self.stack == 'h' else torch.vstack
        temp = func(list([m(x) for m in self.modules]))
        return temp
    
    def eval(self) -> Self:
        for m in self.modules:
            m.eval()
        return super().eval()
    
    def train(self, mode: bool = True) -> Self:
        for m in self.modules:
            m.train(mode=mode)
        return super().train(mode)

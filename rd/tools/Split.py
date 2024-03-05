"""
File: Split.py
Author: Sebastian Hönel
"""

from torch import nn, Tensor
import torch

class Split(nn.Module):
    """

    """
    def __init__(self, *args: nn.Module, **kwargs) -> None:
        super().__init__()

        self.modules = list(args)
        if len(self.modules) == 0:
            raise Exception('Need at least one module.')
        
    def forward(self, x: Tensor) -> Tensor:
        temp = torch.hstack(list([m(x) for m in self.modules]))
        return temp

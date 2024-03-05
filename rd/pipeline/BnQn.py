import numpy as np
import torch
from torch import nn, cuda, device, Tensor
from rd.tools.Split import Split


import numpy as np
import torch
from torch import nn, Tensor, empty, fill, device, nan
from typing import Callable, Self
from math import sqrt




from rd.bpn.bpitnorm.modules.BatchPitNormalization import BatchPitNorm1d



class BnQn(nn.Module):

    def __init__(self, file: str, dev: device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.raw: np.ndarray = np.load(file=file)
        self.raw = self.raw.reshape((self.raw.shape[0], np.prod(self.raw.shape[1:])))
        
        assert len(self.raw.shape) == 2
        self.num_feats = self.raw.shape[1]

        self.bn1 = nn.BatchNorm1d(num_features=self.num_feats, device=dev)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_feats, device=dev)

        self.model = nn.Sequential(
            Split(
                self.bn1,
                nn.Sequential(
                    BatchPitNorm1d(num_features=self.num_feats, dev=dev, num_pit_samples=1000, take_num_samples_when_full=0, normal_backtransform=False, trainable_bandwidths=False),
                    self.bn2
                )
            )
        )

        self._init_bns()
    

    def _init_bns(self):
        temp = torch.tensor(self.raw)
        means = torch.mean(temp, dim=0)
        sds = torch.std(temp, dim=0)

        bn1_params = list(self.bn1.parameters())
        bn1_params[0].data = nn.parameter.Parameter(sds)
        bn1_params[1].data = nn.parameter.Parameter(means)


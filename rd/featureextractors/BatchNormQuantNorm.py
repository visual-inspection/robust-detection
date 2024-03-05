import numpy as np
import torch
from torch import nn, cuda, device, Tensor
from rd.tools.Split import Split


import numpy as np
import torch
from torch import nn, Tensor, empty, fill, device, nan
from typing import Callable, Self, Self, Sequence, Optional
from math import sqrt
from pathlib import Path




from rd.bpn.bpitnorm.modules.BatchPitNormalization import BatchPitNorm1d



class BnQn1d(nn.Module):
    """
    This is meant to be a generic feature extractor that sits atop a regular feature
    extractor or even atop another AE. The purpose is the following:
    
    - Split the input data into two identical copies
    - The first copy goes trough ordinary batch normalization. We'll read the parameters
      for mean and sd directly from the (or learn it if online)
    - The second copy first goes through quantile normalization as realized through
      BatchProbabilityIntegralTransformNormalization. Again, we might fill this layer
      with data or have it learn online.
      - The second copy then gets passed through another ordinary batch normalization
        of its own. The idea is to have two outputs that are in the same range (that is,
        data with mean=~0 and sd=~1).
    """
    def __init__(self, input_shape: Sequence[int], load_data: Optional[Path], dev: device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_feats = np.prod(np.array(input_shape))
        self.online_learning = load_data is None

        # We'll flatten the data and not assume any more dimensions.
        # This is the first copy that goes through regular BN.
        # If online learning, then we need affine=True
        self.bn1 = nn.BatchNorm1d(num_features=self.num_feats, affine=self.online_learning)

        # Here's the second copy that'll be followed by another ordinary BN.
        self.qn = BatchPitNorm1d(num_features=self.num_feats, num_pit_samples=2500, take_num_samples_when_full=0, dev=dev, normal_backtransform=False, trainable_bandwidths=False)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_feats, affine=self.online_learning)

        self.model = Split(
            self.bn1,
            nn.Sequential(self.qn, self.bn2),
            stack='v') # Stack the resulting tensors vertically!
        
        if isinstance(load_data, Path):
            temp: np.ndarray = np.load(file=load_data)
            # Reshape:
            temp = temp.reshape((temp.shape[0], self.num_feats))
            self._init_from_data(data=torch.tensor(temp))
    
    def _init_from_data(self, data: Tensor):
        assert len(data.shape) == 2
        # The first BN is straightforward, get mean/sd from data:
        sm = torch.std_mean(input=data, dim=0)
        bn1_params = list(self.bn1.parameters())
        # 0 is sd, 1 is mean
        bn1_params[0].data = nn.parameter.Parameter(sm[0])
        bn1_params[1].data = nn.parameter.Parameter(sm[1])


        # Next, we fill the Pit normalizer:
        self.qn.fill(data=data)
        # .. and transform the data so that it can be used to
        # initialize the adjacent BN:
        res = self.qn.forward(x=data)

        # Init the 2nd BN:
        sm = torch.std_mean(input=res, dim=0)
        bn2_params = list(self.bn2.parameters())
        bn2_params[0].data = nn.parameter.Parameter(sm[0])
        bn2_params[1].data = nn.parameter.Parameter(sm[1])
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x=x)
    
    def eval(self) -> Self:
        self.model.eval()
        return super().eval()
    
    def train(self, mode: bool = True) -> Self:
        self.model.train(mode=mode)
        return super().train(mode)

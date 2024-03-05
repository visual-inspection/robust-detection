import torch
from torch import nn, Tensor, empty, fill, device, nan
from typing import Callable, Self, Sequence
import numpy as np
from numpy import ndarray
from math import sqrt, exp

def sigmoid(x: float) -> float:
    return 1.0 / (1 + exp(-x))









class QuantNorm1D_new(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_pit_samples: int, take_num_samples_when_full: int, dev: device, normal_backtransform: bool = True, trainable_bandwidths: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        assert self.batch_size <= num_pit_samples
        self.num_pit_samples = num_pit_samples
        assert take_num_samples_when_full <= self.batch_size
        self.take_num_samples_when_full = take_num_samples_when_full
        self.num_features = np.prod(np.array(input_shape[1:]))

        self.trainable_bandwidths = trainable_bandwidths
        if trainable_bandwidths:
            self.bw = torch.nn.Parameter(data=torch.rand(size=(1, self.num_features,)), requires_grad=True).to(device=dev)
        else:
            self.bw = fill(input=torch.empty(size=(1, self.num_features,)), value=nan)

        self.size = 0 # Here we keep track of by how much the values are filled

        values = empty(size=(self.num_pit_samples, self.num_features), device=dev)
        values = fill(input=values, value=nan)
        self.register_buffer(name='cdf_data', persistent=True, tensor=values)

        self.normal_backtransform = normal_backtransform
    
    @property
    def values(self) -> Tensor:
        return self.get_buffer(target='cdf_data')
    
    @property
    def is_full(self) -> bool:
        return self.size == self.num_pit_samples
    
    @property
    def capacity_left(self) -> int:
        return self.num_pit_samples - self.size
    
    def fill(self, batch: Tensor) -> Self:
        batch_size = batch.shape[0]
        assert batch_size <= self.batch_size
        cap_left = self.capacity_left

        if cap_left >= batch_size:
            # Full take, store the entire batch's data in our values.
            self.values[self.size:(self.size + batch_size)] = batch
            self.size += batch_size
        elif cap_left > 0:
            # Take the first elements, then call this method again with remainder of batch.
            self.values[self.size:self.num_pit_samples] = batch[0:cap_left]
            self.size += cap_left
            # Choose accordingly for the remaining values:
            self.fill(batch=batch[cap_left:batch_size])
        else:
            if self.take_num_samples_when_full == 0:
                return self
            # No capacity left.
            use_batch_indexes = torch.randperm(n=min(batch_size, self.take_num_samples_when_full))
            use_values_indexes = torch.randperm(n=self.num_pit_samples)[0:min(batch_size, self.take_num_samples_when_full)]
            self.values[use_values_indexes] = batch[use_batch_indexes]

        return self
    
    @staticmethod
    def standard_normal_cdf(x: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.special.erf(x / sqrt(2.0)))
    
    @staticmethod
    def standard_normal_ppf(x: Tensor) -> Tensor:
        # Values smaller/larger than the following will return (-)inf,
        # so we gotta clip them.
        _min = 9e-8
        _max = 1.0 - _min
        x = torch.clip(input=x, min=_min, max=_max)
        res = sqrt(2.0) * torch.special.erfinv(2.0 * x - 1.0)
        assert not torch.any(torch.isnan(res)) and not torch.any(torch.isinf(res))
        return res
    

    def make_cdf(self, data: Tensor, bw: float) -> Callable[[float], float]:
        num_samples = data.shape[0]
        if not self.trainable_bandwidths:
            q25 = torch.quantile(input=data, q=.25, dim=0)
            q75 = torch.quantile(input=data, q=.75, dim=0)
            IQR = q75 - q25
            bw = 0.9 * torch.min(data.std(), IQR / 1.34) * float(num_samples)**(-.2)
        else:
            bw = torch.sigmoid(bw) # Ensure it's positive.
        return lambda use_x: 1.0 / num_samples * torch.sum(QuantNorm1D_new.standard_normal_cdf((use_x - data) / bw))
    

    def process_merged(self, all_data: Tensor, bandwidths: Tensor, num_data: int, samples_offset: int, num_samples: int) -> Tensor:
        data_cdf = all_data[0:num_data]
        data_sample = all_data[samples_offset:(samples_offset + num_samples)]

        cdf = self.make_cdf(data=data_cdf, bw=bandwidths)
        vcdf = torch.vmap(cdf, in_dims=0, out_dims=0)

        return vcdf(data_sample)
    
    def forward(self, x: Tensor) -> Tensor:
        # Then we'll copy the current batch into it, too:
        batch_size = x.shape[0]
        # Must be asserted because we have limited space only.
        assert batch_size <= self.batch_size
        # First let's fill up the buffered values for the underlying CDFs.
        if self.training:
            self.fill(batch=x)
        else:
            assert self.size > 0, 'Cannot compute forward pass without sample for the integral transform.'

        all_data = torch.vstack((self.values, x))
        assert all_data.shape[0] == self.num_pit_samples + batch_size
        vfunc = torch.vmap(self.process_merged, in_dims=1, out_dims=1)
        result = vfunc(
            all_data, self.bw, num_data=self.size,
            samples_offset=self.num_pit_samples, num_samples=batch_size)

        if self.normal_backtransform:
            result = QuantNorm1D_new.standard_normal_ppf(x=result)
        else:
            result -= 0.5
        return result



dev = 'cuda'
num_feats = 1000
num_samples = 64
cdf_samples = 3000

q1d = QuantNorm1D_new(input_shape=(num_samples, num_feats), num_pit_samples=cdf_samples, take_num_samples_when_full=16, normal_backtransform=True, trainable_bandwidths=True, dev=dev)


x: Tensor = torch.rand(size=(num_samples, num_feats)).to(dev)
res = q1d.forward(x=x)
q1d.eval()
print(5)


def test_filling():
    dev = 'cuda'
    q1d = QuantNorm1D_new(input_shape=(32,10), num_pit_samples=100, take_num_samples_when_full=10, dev=dev, normal_backtransform=False)

    for _ in range(1000):
        q1d.fill(torch.rand(size=(24,10)).to(dev))
    
    return 5

#test_filling()



def test_qnorm(org_x: Tensor, cdf_data: Tensor, vmap_result: Tensor):
    org_x: ndarray = org_x.cpu().numpy()
    vmap_result: ndarray = vmap_result.detach().cpu().numpy()
    cdf_data: ndarray = cdf_data.cpu().numpy()
    from scipy.special import erf
    from scipy.stats.distributions import norm

    normal_dist = norm(loc=0.0, scale=1.0)

    std_normal_cdf = normal_dist.cdf    
    std_normal_ppf = normal_dist.ppf

    def kde_cdf(data: ndarray, bw: float = None) -> Callable[[float], float]:
        if bw is None:
            q25 = np.quantile(a=data, q=.25)
            q75 = np.quantile(a=data, q=.75)
            IQR = q75 - q25
            bw = 0.9 * min(data.std(), IQR / 1.34) * float(data.size)**(-.2)
        return lambda x_val: 1.0 / data.size * np.sum(std_normal_cdf((x_val - data) / bw))
    
    _min = 9e-8
    _max = 1.0 - _min
    
    # Let's check this feature-wise.
    for feat_idx in range(num_feats):
        bw: float = None
        if q1d.trainable_bandwidths:
            bw = sigmoid(q1d.bw[0, feat_idx].item())
        cdf = kde_cdf(data=cdf_data[:, feat_idx], bw=bw)

        for sample_idx in range(num_samples):
            val_expected = cdf(org_x[sample_idx, feat_idx])
            if q1d.normal_backtransform:
                val_expected = std_normal_ppf(min(_max, max(_min, val_expected)))
            else:
                val_expected -= 0.5
            val_actually = vmap_result[sample_idx, feat_idx]
            if np.abs(val_actually - val_expected) > 2e-5: # 1 / 5000
                raise Exception((sample_idx, feat_idx, np.abs(val_actually - val_expected)))



test_qnorm(org_x=x, cdf_data=x, vmap_result=res)
print(5)
import torch
from torch import nn, Tensor, empty, fill, device, nan, is_tensor, mean, std_mean, ones
from typing import Callable
import numpy as np
from numpy import ndarray
from math import sqrt



dev = 'cuda'
num_feats = 1000
num_samples = 64





def std_normal_cdf(x):
    return 0.5 * (1.0 + torch.special.erf(x / sqrt(2.0)))


def make_cdf(data: Tensor) -> Callable[[float], float]:
    num_samples = data.shape[0]
    q25 = torch.quantile(input=data, q=.25, dim=0)
    q75 = torch.quantile(input=data, q=.75, dim=0)
    IQR = q75 - q25
    bw = 0.9 * torch.min(data.std(), IQR / 1.34) * float(num_samples)**(-.2)
    return lambda use_x: 1.0 / num_samples * torch.sum(std_normal_cdf((use_x - data) / bw))


def process_x_and_feat(x_feature: Tensor, data: Tensor, use_num_samples: int) -> Tensor:
    indices = torch.tensor(range(use_num_samples)).to(dev)
    x_non_nan = torch.index_select(input=x_feature, dim=0, index=indices)

    cdf = make_cdf(data=data)
    vcdf = torch.vmap(cdf, in_dims=0, out_dims=0)

    return vcdf(x_non_nan)


# 10 samples with num_feats features
x: Tensor = torch.rand(size=(num_samples, num_feats)).to(dev)
cdf_data: Tensor = torch.rand(size=(1000, num_feats)).to(dev)

x_pad = empty(size=cdf_data.shape).to(dev)
x_pad = fill(x_pad, nan)
x_pad[0:num_samples] = x

vec = torch.vmap(process_x_and_feat, in_dims=1, out_dims=1, chunk_size=5000)
temp = vec(x_pad, cdf_data, use_num_samples=num_samples)
print(temp.shape)



def test_qnorm(org_x: Tensor, cdf_data: Tensor, vmap_result: Tensor):
    org_x: ndarray = org_x.cpu().numpy()
    vmap_result: ndarray = vmap_result.cpu().numpy()
    cdf_data: ndarray = cdf_data.cpu().numpy()
    from scipy.special import erf

    def std_normal_cdf(x: float):
        return 0.5 * (1.0 + erf(x / np.sqrt(2)))

    def kde_cdf(data: ndarray) -> Callable[[float], float]:
        q25 = np.quantile(a=data, q=.25)
        q75 = np.quantile(a=data, q=.75)
        IQR = q75 - q25
        bw = 0.9 * min(data.std(), IQR / 1.34) * float(data.size)**(-.2)
        return lambda x_val: 1.0 / data.size * np.sum(std_normal_cdf((x_val - data) / bw))
    
    # Let's check this feature-wise.
    for feat_idx in range(num_feats):
        cdf = kde_cdf(data=cdf_data[:, feat_idx])

        for sample_idx in range(num_samples):
            val_expected = cdf(org_x[sample_idx, feat_idx])
            val_actually = vmap_result[sample_idx, feat_idx]
            if np.abs(val_actually - val_expected) > 2e-5: # 1 / 5000
                raise Exception((sample_idx, feat_idx, np.abs(val_actually - val_expected)))



test_qnorm(org_x=x, cdf_data=cdf_data, vmap_result=temp)
print(5)



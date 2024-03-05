import torch
from torch import nn, Tensor, empty, fill, device, nan, is_tensor, mean, std_mean, ones
from typing import Callable
import numpy as np
from numpy import ndarray
from math import sqrt



dev = 'cuda'
num_feats = 10000
num_samples = 64
num_cdf = 3000





def std_normal_cdf(x):
    return 0.5 * (1.0 + torch.special.erf(x / sqrt(2.0)))


def make_cdf(data: Tensor) -> Callable[[float], float]:
    num_samples = data.shape[0]
    q25 = torch.quantile(input=data, q=.25, dim=0)
    q75 = torch.quantile(input=data, q=.75, dim=0)
    IQR = q75 - q25
    bw = 0.9 * torch.min(data.std(), IQR / 1.34) * float(num_samples)**(-.2)
    return lambda use_x: 1.0 / num_samples * torch.sum(std_normal_cdf((use_x - data) / bw))



# num_samples samples with num_feats features
x: Tensor = torch.rand(size=(num_samples, num_feats)).to(dev)
cdf_data: Tensor = torch.rand(size=(num_cdf, num_feats)).to(dev)
all_data = torch.concat((cdf_data, x))


def process_merged(all_data: Tensor, num_data: int, num_samples: int) -> Tensor:
    data_cdf = all_data[0:num_data]
    data_sample = all_data[num_data:(num_data + num_samples)]

    cdf = make_cdf(data=data_cdf)
    vcdf = torch.vmap(cdf, in_dims=0, out_dims=0)

    return vcdf(data_sample)


temp = torch.vmap(process_merged, in_dims=1, out_dims=1, chunk_size=5000)(all_data, num_data=num_cdf, num_samples=num_samples)
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
            if np.abs(val_actually - val_expected) > 1e-5: # 1 / 5000
                raise Exception((sample_idx, feat_idx, np.abs(val_actually - val_expected)))



test_qnorm(org_x=x, cdf_data=cdf_data, vmap_result=temp)
print(5)



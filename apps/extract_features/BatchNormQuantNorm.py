###########################################################################
# Paste the following 3 lines at the beginning of each app to make it work.
from sys import path; from pathlib import Path; apps = Path(__file__);
while apps.parent.name != 'apps': apps = apps.parent;
path.append(str(apps.parent.parent)); del apps;
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER;
###########################################################################

"""
Author: Sebastian Hönel

In this app, we init the BnQn1d from data instead of training it. Then, we can
use it as a rather static feature extractor.
"""

import torch
from torch import cuda, tensor
from rd.featureextractors.BatchNormQuantNorm import BnQn1d
from pathlib import Path
from numpy import load, save


if __name__ == '__main__':
    dev = 'cuda' if cuda.is_available() else 'cpu'

    file = DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/train-good-CNxV2_DepthAE.npy')
    temp = tensor(data=load(file=file)).to(device=dev)
    b = BnQn1d(input_shape=temp.shape[1:], dev=dev, load_data=file)
    torch.save(obj=b.state_dict(), f=MODELS_FOLDER.joinpath(f'./{BnQn1d.__name__}_DepthAE_ConvNeXt_V2.torch'))

    for stem in ['train-good', 'test-good', 'test-defect']:
        # Let's inference all the files with the same model as trained on good data.
        file = DATA_FOLDER.joinpath(f'./mvtec_cable_eq_features-scoring/{stem}-CNxV2_DepthAE.npy')
        temp = tensor(data=load(file=file)).to(device=dev)
        res = b.forward(temp).detach().cpu().numpy()
        save(file=DATA_FOLDER.joinpath(f'./mvtec_cable_eq_features-scoring/{stem}-CNxV2_DepthAE_BnQn1d.npy'), arr=res)

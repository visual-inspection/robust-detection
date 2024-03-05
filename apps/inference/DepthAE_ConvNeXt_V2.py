###########################################################################
# Paste the following 3 lines at the beginning of each app to make it work.
from sys import path; from pathlib import Path; apps = Path(__file__);
while apps.parent.name != 'apps': apps = apps.parent;
path.append(str(apps.parent.parent)); del apps;
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER;
###########################################################################

"""
Author: Sebastian Hönel
"""

import torch
from torch import cuda, device, Tensor
from rd.data.ConvNeXt_V2 import Dataset_ConvNeXt_V2
from rd.autoencoders.ConvNeXt_V2_DepthAE import DepthAE, InferenceDepthAE
from numpy import save



if __name__ == '__main__':

    inference_dataset = Dataset_ConvNeXt_V2(
        scaler=None,
        move_channels_first=True,
        file=DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/test-defect-CNxV2.npy'))

    dev: device = 'cuda' if cuda.is_available() else 'cpu'
    model = DepthAE(dev=dev)
    model.load_state_dict(state_dict=torch.load(MODELS_FOLDER.joinpath(f'./{DepthAE.__name__}_ConvNeXt_V2.torch')))
    idae = InferenceDepthAE(model=model)
    temp: Tensor = idae.forward(torch.tensor(inference_dataset.raw).to(dev))

    result = temp.detach().cpu().numpy()
    save(file=DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/test-defect-CNxV2_DepthAE.npy'), arr=result)

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

from pathlib import Path
import numpy as np


if __name__ == '__main__':
    mvtec = DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/')
    train_good = mvtec.joinpath('./train-good-CNxV2_DepthAE_BnQn1d.npy')
    test_good = mvtec.joinpath('./test-good-CNxV2_DepthAE_BnQn1d.npy')
    test_defect = mvtec.joinpath('./test-defect-CNxV2_DepthAE_BnQn1d.npy')

    temp = np.load(train_good)
    pass

"""
Here we try to apply a PCA using JAX.
Then, we train a random forest on the dim-reduced data and show some results.
This file uses features as extracted from ConvNeXt V2 (approx. 700k features).
In this current configuration, we get about a 66% accuracy with 0.30 Kappa (bad).
"""
from sys import path
from pathlib import Path

apps = Path(__file__)
while apps.parent.name != 'apps':
    apps = apps.parent
path.append(str(apps.parent.parent))
del apps
from apps import APPS_FOLDER, DATA_FOLDER, MODELS_FOLDER

from os import environ

environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(jax.local_devices())
jax.config.update('jax_platform_name', 'gpu')
d = jax.devices('gpu')

import numpy as np
from spax import PCA_m

pca = PCA_m(devices=jax.devices('gpu'), N=256)

mvtec = DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/')
train_file = mvtec.joinpath('./train-good-CNxV2.npy')
test_defect = mvtec.joinpath('./test-defect-CNxV2.npy')
test_good = mvtec.joinpath('./test-good-CNxV2.npy')

train = np.load(file=train_file)
train = train.transpose()
pca.fit(data=train)

train_t = pca.transform(train)

test_good = np.load(file=test_good)
test_good = test_good.transpose()
test_defect = np.load(file=test_defect)
test_defect = test_defect.transpose()

test_good_t = pca.transform(test_good)
test_defect_t = pca.transform(test_defect)

np.save(mvtec.joinpath('./train-CNxV2-good-PCA.npy'), train_t)

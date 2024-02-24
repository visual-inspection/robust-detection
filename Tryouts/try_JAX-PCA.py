"""
Here we try to apply a PCA using JAX.
Then, we train a random forest on the dim-reduced data and show some results.
This file uses features as extracted from ConvNeXt V2 (approx. 700k features).
In this current configuration, we get about a 66% accuracy with 0.30 Kappa (bad).
"""

from os import environ
environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update('jax_platform_name', 'cpu')
d = jax.devices('gpu')

import numpy as np
from spax import PCA_m

pca = PCA_m(devices=jax.devices('gpu'), N=10)

train = np.load(file='train-cnxV2-good.npy')
train = train.transpose()
pca.fit(data=train)

train_t = pca.transform(train)

test_good = np.load(file='test-cnxV2-good.npy')
test_good = test_good.transpose()
test_defect = np.load(file='test-cnxV2-defect.npy')
test_defect = test_defect.transpose()

test_good_t = pca.transform(test_good)
test_defect_t = pca.transform(test_defect)

X = np.concatenate([test_good_t, test_defect_t], axis=1).transpose()
Y = np.concatenate([
    np.array([0] * test_good.shape[1]),
    np.array([1] * test_defect.shape[1])])
Y = Y.reshape((Y.shape[0], 1))


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=1337)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train)
X_test_scaled = scaler.transform(X=X_test)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0xbeef)
rf.fit(X=X_train_scaled, y=Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

y_true, y_pred = Y_test, rf.predict(X=X_test_scaled)
for m in [accuracy_score, cohen_kappa_score, confusion_matrix]:
    print(f'{m.__name__}:')
    print(m(y_true, y_pred))

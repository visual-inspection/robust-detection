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

from torch import cuda
from rd.pipeline.BnQn import BnQn


if __name__ == '__main__':
    dev = 'cuda' if cuda.is_available() else 'cpu'
    b = BnQn(file='./BnQn.npy', dev=dev)
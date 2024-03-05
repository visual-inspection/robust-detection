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


from rd.featureextractors.ConvNeXt_V2 import FeatureExtractor_ConvNeXTV2



if __name__ == '__main__':
    fe = FeatureExtractor_ConvNeXTV2(
        # Path to folder with images
        in_folder=DATA_FOLDER.joinpath('./tmp'), # You can store some data here
        # Output-path. Will write file with stemname there and extension .npy or .csv.
        out_folder=DATA_FOLDER.joinpath('./mvtec_cable_eq_features-scoring/'))

    fe.load_images(verbose=True)
    data = fe.extract(swap_channels_last=True)
    fe.save(outfile_stemname='train-good-CNxV2', data=data, save_numpy=True, save_csv=False)

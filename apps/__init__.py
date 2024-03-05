from sys import path
from pathlib import Path

APPS_FOLDER = Path(__file__).parent

DATA_FOLDER = APPS_FOLDER.parent.joinpath('./data')
MODELS_FOLDER = APPS_FOLDER.parent.joinpath('./models')

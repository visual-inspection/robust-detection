from pathlib import Path
from typing import Iterable



class FeatureExtractor():
    """
    This is the mother of all feature extractors. It is meant to provide
    common functionality, as well as a common interface for its sub-classes.
    """
    def __init__(self, in_folder: Path, out_folder: Path) -> None:
        self.in_folder = in_folder
        self.out_folder = out_folder


    def extract_from(self, images: Iterable[Path], save_numpy: bool=True, save_csv: bool=True):
        raise NotImplementedError('Abstract Method.')


    @staticmethod
    def files(folder: str, exts: Iterable[str]=['bmp', 'gif', 'jpg', 'jpeg', 'png']) -> Iterable[Path]:
        return Path(folder).glob(pattern=f'*.{{ {','.join(exts)} }}')

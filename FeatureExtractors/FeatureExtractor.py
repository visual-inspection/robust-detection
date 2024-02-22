from pathlib import Path
from typing import Iterable
from numpy import ndarray, save, transpose
from pandas import DataFrame
from typing import Self



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


    def save(self, images: Iterable[Path], outfile_stemname: str, data: ndarray, save_numpy: bool=True, save_csv: bool=True) -> Self:
        if not (save_csv or save_numpy):
            raise Exception('You should save something.')
        
        if save_numpy:
            outfile_numpy = self.out_folder.joinpath(f'{outfile_stemname}.npy')
            save(file=outfile_numpy, arr=data)
        
        if save_csv:
            outfile_pandas = self.out_folder.joinpath(f'{outfile_stemname}.csv')
            df = DataFrame(transpose(data))
            df.columns = list([f'i_{file.stem}' for file in images])
            df.to_csv(path_or_buf=outfile_pandas, index=False)


    @staticmethod
    def files(folder: str, exts: Iterable[str]=['bmp', 'gif', 'jpg', 'jpeg', 'png']) -> Iterable[Path]:
        return Path(folder).glob(pattern=f'*.{{ {','.join(exts)} }}')

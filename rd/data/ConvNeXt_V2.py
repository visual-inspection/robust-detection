from torch.utils.data import Dataset
from torch import is_tensor
from sklearn.preprocessing import StandardScaler
from numpy import ndarray, load, swapaxes
from pathlib import Path




class Dataset_ConvNeXt_V2(Dataset):
    """
    Loads a dataset that was previously created by using the ConvNeXt_V2 feature extractor.
    """
    def __init__(self, file: str | Path, move_channels_first: bool = True, scaler: StandardScaler = None) -> None:
        """
        :param bool move_channels_first If true, will swap the axes such that the channels come first.
        This is the default for Pytorch.
        :param StandardScaler scaler If given a new instance, it will be fit to the data. If it was previously
        fit, then it is applied to the data.
        """
        super().__init__()
        self.raw: ndarray = load(file=file)
        if isinstance(scaler, StandardScaler):
            # Check if it was fit -> If so, apply (otherwise fit, apply, and store)
            if not hasattr(scaler, 'n_features_in_') or scaler.n_features_in_ == 0:
                # Was not fit before
                scaler.fit(X=self.raw)
            
            self.raw = scaler.transform(X=self.raw)
        
        self.scaler = scaler

        self.raw = self.raw.reshape((self.raw.shape[0], 16, 16, 2816))
        if move_channels_first:
            # 16x16x2816 (channels last)  -->>  2816x16x16 (channels first)
            self.raw = swapaxes(swapaxes(self.raw, 3, 2), 2, 1)
    
    def __len__(self) -> int:
        return self.raw.shape[0]
    
    def __getitem__(self, index) -> ndarray:
        if is_tensor(index):
            index = index.tolist()
        
        # Since we want to train an auto-encoder, X and Y are equal.
        temp = self.raw[index]
        return {'input': temp, 'output': temp}

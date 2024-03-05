from transformers import AutoImageProcessor, ConvNextV2Model
from pathlib import Path
from PIL import Image
from numpy import ndarray
import torch
import numpy as np


from FeatureExtractor import FeatureExtractor


class FeatureExtractor_ConvNeXTV2(FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path) -> None:
        super().__init__(in_folder, out_folder)

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-huge-22k-512").to(self.device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-huge-22k-512")


    def extract(self, swap_channels_last: bool = True) -> ndarray:
        results: list = []
        with torch.no_grad():
            for file in self.images:
                print(f'Processing file: {file.name}')
                inputs = self.image_processor(Image.open(fp=file), return_tensors='pt').to(self.device)

                # BaseModelOutputWithPoolingAndNoAttention
                temp = self.model(**inputs).last_hidden_state.cpu().numpy()
                if swap_channels_last:
                    # Channels First  -->> Channels Last
                    # 1 x 2816 x 16 x 16  -->>  1 x 16 x 2816 x 16  -->>  1 x 16 x 16 x 2816
                    temp = np.swapaxes(np.swapaxes(temp, 1, 2), 2, 3)
                # -->> 1 x 720896
                temp = temp.flatten()
                results.append(temp)
        
        return np.stack(results)


if __name__ == '__main__':
    fe = FeatureExtractor_ConvNeXTV2(
        in_folder=Path('/tmp/input'),
        out_folder=Path('/tmp/output'))

    fe.load_images(verbose=True)
    data = fe.extract(swap_channels_last=True)
    fe.save(outfile_stemname='train-good-CNv2', data=data, save_numpy=True, save_csv=True)

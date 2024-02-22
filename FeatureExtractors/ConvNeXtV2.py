from transformers import AutoImageProcessor, ConvNextV2Model
from pathlib import Path
from PIL import Image
import torch
import numpy as np


from .FeatureExtractor import FeatureExtractor


class FeatureExtractor_ConvNeXTV2(FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path) -> None:
        super().__init__(in_folder, out_folder)

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-huge-22k-512").to(self.device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-huge-22k-512")


    def extract_from(self, images: np.Iterable[Path], outfile_stemname: str, save_numpy: bool = True, save_csv: bool = True, swap_channels_last: bool = True):
        results: list = []
        with torch.no_grad():
            for file in images:
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
        
        all_features = np.stack(results)
        return self.save(images=images, outfile_stemname=outfile_stemname, data=all_features, save_numpy=save_numpy, save_csv=save_csv)

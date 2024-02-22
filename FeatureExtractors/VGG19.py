from .FeatureExtractor import FeatureExtractor
from keras_core.applications.vgg19 import VGG19, preprocess_input
from keras_core.layers import Layer, Flatten, Concatenate, Conv2D, MaxPool2D
from keras_core.models import Model
from keras_core.activations import linear
from keras_core.preprocessing.image import load_img, img_to_array
from pathlib import Path
from numpy import ndarray
from typing import Self
import numpy as np
import pandas as pd



class FeatureExtractor_VGG19(FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path) -> None:
        super().__init__(in_folder, out_folder)
        self.model = VGG19(include_top=False)

        self._conv_layers: list[Conv2D] = []
        self._max_layers: list[MaxPool2D] = []
        for layer in self.model.layers:
            layer.trainable = False
            if 'conv' in layer.name:
                self._conv_layers.append(layer)
            if 'pool' in layer.name:
                self._max_layers.append(layer)
        
        self._output_layers: list[Layer] = [self._conv_layers[-1]]
        self._use_linear_cnn_act = True
    
    @property
    def conv_layers(self) -> list[Conv2D]:
        return self._conv_layers.copy()
    
    @property
    def max_layers(self) -> list[MaxPool2D]:
        return self._max_layers.copy()

    @property
    def output_layers(self) -> list[Layer]:
        return self._output_layers
    
    @output_layers.setter
    def output_layers(self, use_layers: list[Layer]) -> Self:
        self._output_layers = use_layers
        return self
    
    @property
    def use_linear_cnn_act(self) -> bool:
        return self.use_linear_cnn_act
    
    @use_linear_cnn_act.setter
    def use_linear_cnn_act(self, use: bool) -> Self:
        self._use_linear_cnn_act = use
        return self
    
    def extract_from(self, images: np.Iterable[Path], outfile_name: str, save_numpy: bool = True, save_csv: bool = True):
        model = self.model if len(self.output_layers) == 0 else Model(self.model.inputs, list([l.output for l in self.output_layers]))

        if not (save_csv or save_numpy):
            raise Exception('You should save something.')

        images = list(images)

        image_inputs: list[ndarray] = []
        for file in images:
            filename = str(file.resolve())
            print(f'Reading file: {file.name}')
            img = load_img(path=filename, target_size=(224, 224))
            img = img_to_array(img=img)
            img = preprocess_input(x=img)
            image_inputs.append(img)

        all_images_np = np.stack(image_inputs)
        results = model.predict(all_images_np)
        if not isinstance(results, list):
            results = [results]

        all_features = Concatenate()(list([Flatten()(r) for r in results]))
        
        if save_numpy:
            outfile_numpy = self.out_folder.joinpath(f'{outfile_name}.npy')
            np.save(file=outfile_numpy, arr=all_features)
        
        if save_csv:
            outfile_pandas = self.out_folder.joinpath(f'{outfile_name}.csv')
            df = pd.DataFrame(np.transpose(all_features))
            df.columns = list([f'i_{file.stem}' for file in images])
            df.to_csv(path_or_buf=outfile_pandas, index=False)

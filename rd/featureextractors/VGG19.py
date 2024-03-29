"""
File: VGG19.py
Author: Sebastian Hönel
"""

# TODO: Replace implementation with Pytorch if desired.


from .FeatureExtractor import FeatureExtractor
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Layer, Flatten, Concatenate, Conv2D, MaxPool2D
from keras.models import Model
from keras.activations import linear
from keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
from numpy import ndarray
from typing import Self
import numpy as np



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
        if not isinstance(use_layers, list):
            raise Exception("The argument output_layers must be a list.")
        self._output_layers = use_layers
        return self
    
    @property
    def use_linear_cnn_act(self) -> bool:
        """
        This only has an effect if a custom list of output layers is
        used. If so, then the activation on those layers which are Conv2D
        will be set to linear instead. Other layers are not affected.
        Note that this option is initialized to ``True`` by default.
        """
        return self._use_linear_cnn_act
    
    @use_linear_cnn_act.setter
    def use_linear_cnn_act(self, use: bool) -> Self:
        self._use_linear_cnn_act = use
        return self
    
    def extract(self) -> ndarray:
        output_layers = self.output_layers
        if self.use_linear_cnn_act:
            for layer in filter(lambda layer: isinstance(layer, Conv2D), output_layers):
                layer.activation = linear

        model = self.model if len(output_layers) == 0 else Model(self.model.inputs, list([l.output for l in output_layers]))

        image_inputs: list[ndarray] = []
        for file in self.images:
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

        return Concatenate()(list([Flatten()(r) for r in results])).cpu().numpy()

from typing import Tuple
import numpy as np

class Normalize:
    def __init__(self, mean: Tuple, std: Tuple) -> None:
        self.mean = mean
        self.std = std

    def transform(self, image: np.ndarray) -> np.ndarray:
        assert image.shape[-1] == 3, "Expecting RGB image"
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

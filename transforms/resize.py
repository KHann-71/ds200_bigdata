from typing import Tuple
import numpy as np
from PIL import Image

class Resize:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def transform(self, image: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(image.astype(np.uint8), mode='RGB')
        image_resized = image_pil.resize(self.size)
        return np.array(image_resized)

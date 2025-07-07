from typing import List
import numpy as np

class Transforms:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def transform(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t.transform(image)
        return image

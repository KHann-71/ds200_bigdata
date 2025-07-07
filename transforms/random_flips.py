import numpy as np

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            image = np.fliplr(image)
        return image

class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            image = np.flipud(image)
        return image

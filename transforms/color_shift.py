import numpy as np

class ColorShift:
    def __init__(self, r_shift: int, g_shift: int, b_shift: int, p: float = 0.5) -> None:
        self.r = r_shift
        self.g = g_shift
        self.b = b_shift
        self.p = p

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            shifted = np.zeros_like(image)
            shifted[:, :, 0] = np.roll(image[:, :, 0], self.r, axis=0)
            shifted[:, :, 1] = np.roll(image[:, :, 1], self.g, axis=1)
            shifted[:, :, 2] = np.roll(image[:, :, 2], self.b, axis=0)
            return shifted
        return image

import numpy as np


def make_array_divisible_by(image: np.array, div_factor: int):
    if not image.ndim in [3, 4]:
        raise ValueError("Expected a 3D or 4D array as input")

    height, width = image.shape[-3: -1]
    rows = height // div_factor + (1 if height % div_factor else 0)
    cols = width // div_factor + (1 if width % div_factor else 0)

    padding = [(0, rows * div_factor - height), (0, cols * div_factor - width), (0, 0)]
    if image.ndim == 4:
        padding.insert(0, (0, 0))
    image = np.pad(image, padding, mode='constant', constant_values=0)
    
    return image

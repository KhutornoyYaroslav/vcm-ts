import numpy as np


def pad_image(image: np.array, factor: int):
    height, width, _ = image.shape
    rows = height // factor + (1 if height % factor else 0)
    cols = width // factor + (1 if width % factor else 0)

    image = np.pad(image, [(0, rows * factor - height), (0, cols * factor - width), (0, 0)],
                   mode='constant', constant_values=0)
    
    return image
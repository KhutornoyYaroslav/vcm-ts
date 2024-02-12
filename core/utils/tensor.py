import torch
import numpy as np


def channel_to_space(x, out_rows: int, out_cols: int): # BCHW -> BCHW
    assert out_rows * out_cols == x.shape[1]

    if type(x) == torch.Tensor:
        result = torch.zeros(size=(x.shape[0], 1, out_rows * x.shape[2], out_cols * x.shape[3]), dtype=x.dtype)
    else:
        result = np.zeros(size=(x.shape[0], 1, out_rows * x.shape[2], out_cols * x.shape[3]), dtype=x.dtype)

    for i in range(out_rows):
        for j in range(out_cols):
            result[:, 0, i * x.shape[2]:(i + 1) * x.shape[2], j * x.shape[3]:(j + 1) * x.shape[3]] = x[:, i * out_cols + j, :, :]

    return result


def space_to_channel(x, tile_width:int, tile_height:int): # BCHW -> BCHW
    height, width = x.shape[2:4]
    rows, cols = height // tile_height, width // tile_width
    assert rows * tile_height == height and cols * tile_width == width

    result = []
    for i in range(rows):
        for j in range(cols):
            result.append(x[:, 0, i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width])

    return np.stack(result, axis=1) if type(x) == np.ndarray else torch.stack(result, dim=1)

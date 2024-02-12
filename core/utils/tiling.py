import numpy as np


def frame_to_tiles(frame: np.array, tile_size: int, tile_padding: int = 0) -> np.array:
    # Prepare expanded frame
    ts = tile_size - 2 * tile_padding
    assert ts > 0

    height, width, _ = frame.shape
    rows = height // ts + (1 if height % ts else 0)
    cols = width // ts + (1 if width % ts else 0)

    padded_frame = np.pad(frame, [(tile_padding, rows * ts - height + tile_padding),
                                  (tile_padding, cols * ts - width + tile_padding),
                                  (0, 0)],
                          mode='reflect') #, constant_values=0)

    # Split to tiles
    tiles = []
    for y in range(0, rows * ts, ts):
        for x in range(0, cols * ts, ts):
            tile = padded_frame[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

    return np.array(tiles)


def tiles_to_frame(tiles: np.array, frame_h: int, frame_w: int, tile_padding: int = 0):
    # Construct padded image
    n, h, w, c = tiles.shape
    pad = 2 * tile_padding
    ts_h = h - pad
    ts_w = w - pad
    rows = frame_h // ts_h + (1 if frame_h % ts_h else 0)
    cols = frame_w // ts_w + (1 if frame_w % ts_w else 0)
    padded_frame = np.zeros(shape=(rows * (h - pad) + pad, cols * (w - pad) + pad, c), dtype=tiles.dtype)

    tile_idx = 0
    for y in range(0, rows * (h - pad), (h - pad)):
        for x in range(0, cols * (w - pad), (w - pad)):
            padded_frame[y:y + h, x:x + w] = tiles[tile_idx]
            tile_idx += 1

    # Remove padding
    result_frame = padded_frame[tile_padding:tile_padding+frame_h, tile_padding:tile_padding+frame_w]

    return result_frame
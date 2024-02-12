import os
import shutil
import argparse
import cv2 as cv
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from glob import glob
from core.utils import dist_util
from core.utils.logger import setup_logger
from core.utils.tiling import frame_to_tiles, tiles_to_frame


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Onnx Export Test')
    parser.add_argument('--model-path', dest='model_path', type=str, default="outputs/ftvsr/export/onnx/FTVSR_n1_t10_c3_h80_w80.onnx",
                        help="Path to onnx model")
    parser.add_argument('--lrs-root', dest='lrs_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/datasets/test_21_short/artifacts/lr_h265",
                        help="Path to low resolution input frames")
    parser.add_argument('--result-root', dest='result_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/datasets/test_21_short/onnx_test_results",
                        help="Path where to save result frames")
    parser.add_argument('--tile-padding', dest='tile_padding', type=int, default=16,
                        help="Padding size for tiling")
    parser.add_argument('--fname-template', dest='fname_template', type=str, default="%05d.png",
                        help="Filename template for result frames")
    args = parser.parse_args()

    # Create logger
    logger = setup_logger("ONNX TEST", dist_util.get_rank())
    logger.info(args)

    # Create onnxruntime model
    logger.info(f"Onnx session running on {ort.get_device()} device")
    ort_session = ort.InferenceSession(args.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Get model input shape
    input_shape = ort_session.get_inputs()[0].shape
    assert input_shape[3] == input_shape[4]
    chunk_size = input_shape[1]
    tile_size = input_shape[3]

    # Scan input low resolution frames
    lr_filelist = sorted(glob(os.path.join(args.lrs_root, "*")))
 
    # Create result dir
    shutil.rmtree(args.result_root, ignore_errors=True)
    os.makedirs(args.result_root, exist_ok=True)

    # Process each group of frames
    cnt = 0
    pbar = tqdm(total=len(lr_filelist))
    while lr_filelist:
        lr_chunk, lr_filelist = lr_filelist[:chunk_size], lr_filelist[chunk_size:]

        # Split frames to tiles
        w_orig, h_orig = 0, 0
        lr_chunk_tiles = []
        for lr_file in lr_chunk:
            lr_frame = cv.imread(lr_file)
            h_orig, w_orig, _ = lr_frame.shape
            lr_frame = cv.cvtColor(lr_frame, cv.COLOR_BGR2RGB)
            lr_tiles = frame_to_tiles(lr_frame, tile_size, args.tile_padding)
            lr_chunk_tiles.append(lr_tiles)
        lr_chunk_tiles = np.array(lr_chunk_tiles) # (T, N, H, W, C)

        # Process each group of tiles
        ftvsr_chunk_tiles = []
        for i in range(lr_chunk_tiles.shape[1]):
            # Prepare input data
            lr_tiles = lr_chunk_tiles[:, i] # (T, H, W, C)
            lr_tiles = lr_tiles.astype(np.float32) / 255.0
            lr_tiles = lr_tiles.transpose(0, 3, 1, 2) # (T, C, H, W)
            lr_tiles = np.expand_dims(lr_tiles, axis=0) # (N, T, C, H, W)

            # Infer model
            ort_inputs = {ort_session.get_inputs()[0].name: lr_tiles}
            ort_outs = ort_session.run(None, ort_inputs)
            ort_out = ort_outs[0]

            # Process model output
            ftvsr_tiles = np.squeeze(ort_out, 0) # (T, C, H, W)
            ftvsr_tiles = ftvsr_tiles.transpose(0, 2, 3, 1) # (T, H, W, C)
            ftvsr_tiles = np.clip(ftvsr_tiles, 0.0, 1.0)
            ftvsr_tiles = 255.0 * ftvsr_tiles
            ftvsr_tiles = ftvsr_tiles.astype(np.uint8)
            ftvsr_chunk_tiles.append(ftvsr_tiles)
        ftvsr_chunk_tiles = np.array(ftvsr_chunk_tiles) # (N, T, 4*H, 4*W, C)

        # Save frames
        for i in range(ftvsr_chunk_tiles.shape[1]):
            ftvsr_tiles = ftvsr_chunk_tiles[:, i]
            ftvsr_frame = tiles_to_frame(ftvsr_tiles, 4 * h_orig, 4 * w_orig, 4 * args.tile_padding)
            ftvsr_frame = cv.cvtColor(ftvsr_frame, cv.COLOR_RGB2BGR)
            img_path = os.path.join(args.result_root, args.fname_template % cnt)
            cv.imwrite(img_path, ftvsr_frame)
            cnt += 1
            pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()

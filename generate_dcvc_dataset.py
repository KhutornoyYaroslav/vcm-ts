import argparse
import math
import os
import shutil
from glob import glob

import cv2 as cv
import numpy as np
from tqdm import tqdm

from core.utils import dist_util
from core.utils.logger import setup_logger, logging
from core.utils.video import get_video_length

_LOGGER_NAME = "DATASETGEN"


def split_frame(frame: np.array,
                tile_size: int):
    # Prepare expanded frame
    height, width, _ = frame.shape
    rows = height // tile_size + (1 if height % tile_size else 0)
    cols = width // tile_size + (1 if width % tile_size else 0)

    padded_frame = np.pad(frame, [(0, rows * tile_size - height), (0, cols * tile_size - width), (0, 0)],
                          mode='constant', constant_values=0)

    # Split frame to tiles
    tiles = []
    for i in range(rows):
        for j in range(cols):
            tile = padded_frame[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
            tiles.append(tile)

    return tiles


def process_frames(files: list,
                   dst_root: str,
                   tile_size: int,
                   seq_length: int,
                   dirname_template: str = "seq_%05d",
                   filename_template: str = "%05d.png"):
    common_index = 0
    frames_in_seq_count = 0
    seqs_amount = math.floor(len(files) / seq_length)

    for file in tqdm(files):
        frame = cv.imread(file)
        tiles = split_frame(frame, tile_size)

        for tile_index, tile in enumerate(tiles):
            seq_id = tile_index * seqs_amount + common_index
            seq_folder = os.path.join(dst_root, dirname_template % seq_id)

            # Write frame
            raw_seq_folder = os.path.join(seq_folder, 'raw')
            os.makedirs(raw_seq_folder, exist_ok=True)
            raw_filename = os.path.join(raw_seq_folder, filename_template % frames_in_seq_count)
            cv.imwrite(raw_filename, tile)

        frames_in_seq_count += 1
        if frames_in_seq_count == seq_length:
            common_index += 1
            frames_in_seq_count = 0


def generate_data(src_root: str,
                  dst_root: str,
                  tile_size: int,
                  seq_length: int,
                  dirname_template: str = "%05d"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Remove output directory
    shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    # Process source folders
    logger.info("Splitting frames to tiles...")
    folders = sorted(glob(os.path.join(src_root, "*")))
    for folder_index, folder in enumerate(tqdm(folders)):
        out_folder = os.path.join(dst_root, dirname_template % folder_index)
        split_frames_to_tiles(src_root, out_folder, tile_size, seq_length)


def generate_data_video(src_video: str,
                        dst_root: str,
                        tile_size: int,
                        seq_length: int,
                        filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Remove output directory
    shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    # Video to frames
    logger.info("Splitting video to frames...")
    video_filename = os.path.splitext(os.path.basename(src_video))[0]
    video_frames_path = os.path.join(dst_root, "temp_frames")
    shutil.rmtree(video_frames_path, ignore_errors=True)
    os.makedirs(video_frames_path, exist_ok=True)
    video_frames_path_template = os.path.join(video_frames_path, filename_template)
    video2frames(src_video, video_frames_path_template)

    # Process frames folder
    logger.info("Splitting frames to tiles...")
    out_folder = os.path.join(dst_root, video_filename + "_data")
    split_frames_to_tiles(video_frames_path, out_folder, tile_size, seq_length)
    shutil.rmtree(video_frames_path, ignore_errors=True)


def split_frames_to_tiles(frames_path: str,
                          out_folder: str,
                          tile_size: int,
                          seq_length: int):
    logger = logging.getLogger(_LOGGER_NAME)
    logger.info("Verify frames shapes...")
    shape = None
    files = sorted(glob(os.path.join(frames_path, '*')))
    for file in tqdm(files):
        frame = cv.imread(file)
        if shape is not None:
            assert frame.shape == shape
        shape = frame.shape

    logger.info("Process frames...")
    os.makedirs(out_folder, exist_ok=True)
    process_frames(files, out_folder, tile_size, seq_length)


def video2frames(src_path: str,
                 out_files: str,
                 countable: bool = False):
    frame_cnt = 0
    cap_encoded = cv.VideoCapture(src_path)

    pbar = tqdm(total=get_video_length(src_path, countable))
    while cap_encoded.isOpened():
        ret, frame = cap_encoded.read()
        if not ret:
            break
        cv.imwrite(out_files % frame_cnt, frame)
        frame_cnt += 1
        pbar.update(1)
    pbar.close()


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Sequences Dataset Generator')
    parser.add_argument('--src-root', dest='src_root', type=str,
                        default="data/huawei/sources/train/test_0/test_0.mp4",
                        help="Path to dataset root directory or to video file")
    parser.add_argument('--dst-root', dest='dst_root', type=str,
                        default="data/huawei/outputs/train/test_0",
                        help="Path where to save result dataset")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=256,
                        help="Size of output frames")
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=16,
                        help="Number of frames in output sequences")
    parser.add_argument('--video-mode', dest='video_mode', required=False, default=True, type=str2bool,
                        help="Choose video or dataset frames as inputs")
    args = parser.parse_args()

    # Create logger
    logger = setup_logger(_LOGGER_NAME, dist_util.get_rank())
    logger.info(args)

    # Generate tile frames from set of frames or video
    if args.video_mode:
        generate_data_video(args.src_root, args.dst_root, args.tile_size, args.seq_length)
    else:
        generate_data(args.src_root, args.dst_root, args.tile_size, args.seq_length)


if __name__ == '__main__':
    main()

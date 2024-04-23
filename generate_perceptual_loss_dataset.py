import argparse
import os
import shutil
from glob import glob

import cv2
import cv2 as cv
from tqdm import tqdm

from core.utils import dist_util
from core.utils.logger import setup_logger, logging

_LOGGER_NAME = "DATASETGEN"


def get_video_length(path, countable=False):
    cap = cv2.VideoCapture(path)
    if countable:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        count = 0
        while cap.isOpened():
            ret, x = cap.read()
            if not ret:
                break
            count += 1
        cap.release()

    return count


def generate_data_video(src_root: str,
                        dst_root: str,
                        resize_factor: int,
                        seq_length: int,
                        filename_template: str = "im%d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Remove output directory
    shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    # Video to frames
    logger.info("Splitting video to frames...")
    source_videos = sorted(glob(os.path.join(src_root, "*.mp4")))
    sequences_dir = os.path.join(dst_root, "sequences")
    sequences_list = os.path.join(dst_root, "train_sequences.txt")
    for source_video_index, source_video in enumerate(source_videos):
        video_dir = os.path.join(sequences_dir, "%05d" % (source_video_index + 1))
        shutil.rmtree(video_dir, ignore_errors=True)
        os.makedirs(video_dir, exist_ok=True)
        video2frames(source_video, video_dir, filename_template, sequences_list, resize_factor, seq_length)


def video2frames(src_video: str,
                 video_dir: str,
                 filename_template: str,
                 sequences_list: str,
                 resize_factor: int,
                 seq_length: int,
                 countable: bool = False,
                 seq_dir_template: str = "%04d"):
    frame_count = 1
    seq_count = 1
    total_frames = int(get_video_length(src_video, countable))
    cap_encoded = cv.VideoCapture(src_video)
    for _ in tqdm(range(total_frames)):
        if frame_count > seq_length:
            seq_count += 1
            frame_count = 1

        if seq_count * seq_length > total_frames:
            break

        if frame_count == 1:
            seq_dir = os.path.join(video_dir, seq_dir_template % seq_count)
            frame_path = os.path.join(seq_dir, filename_template)
            shutil.rmtree(seq_dir, ignore_errors=True)
            os.makedirs(seq_dir, exist_ok=True)
            seq_local_path = video_dir.split("/")[-1] + "/" + seq_dir_template % seq_count + "\n"
            with open(sequences_list, "a") as f:
                f.write(seq_local_path)

        ret, frame = cap_encoded.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        cv.imwrite(frame_path % frame_count, frame)
        frame_count += 1
    cap_encoded.release()


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Sequences Dataset Generator')
    parser.add_argument('--src-root', dest='src_root', type=str,
                        default="data/huawei/outputs/benchmark/mp4",
                        help="Path to dataset root directory with videos")
    parser.add_argument('--dst-root', dest='dst_root', type=str,
                        default="data/huawei_septuplet",
                        help="Path where to save result dataset")
    parser.add_argument('--resize-factor', dest='resize_factor', type=float, default=0.25,
                        help="Resize factor for output images")
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=7,
                        help="Number of frames in output sequences")
    args = parser.parse_args()

    # Create logger
    logger = setup_logger(_LOGGER_NAME, dist_util.get_rank())
    logger.info(args)

    # Generate dataset from video
    generate_data_video(args.src_root, args.dst_root, args.resize_factor, args.seq_length)


if __name__ == '__main__':
    main()

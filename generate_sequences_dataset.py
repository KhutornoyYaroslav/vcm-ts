import math
import os
import torch
import shutil
import argparse
import cv2 as cv
import numpy as np
from torch import nn
from glob import glob
from tqdm import tqdm
from subprocess import call
from core.utils import dist_util
from core.utils.video import get_video_length
from core.utils.logger import setup_logger, logging
from core.modelling.model.ftvsr import FTVSR
from core.modelling.model.yolov6 import YOLOv6Detector
from core.utils.checkpoint import CheckPointer
from core.utils.matlab_imresize import imresize
from core.data.transforms.functional import upsample_as_torch


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
                   resize_factor: int,
                   seq_length: int,
                   dirname_template: str = "seq_%05d",
                   filename_template: str = "%05d.png"):
    common_index = 0
    frames_in_seq_count = 0
    seqs_amount = math.floor(len(files) / seq_length)

    for file in files:
        frame = cv.imread(file)
        tiles = split_frame(frame, tile_size)

        for tile_index, tile in enumerate(tiles):
            seq_id = tile_index * seqs_amount + common_index
            seq_folder = os.path.join(dst_root, dirname_template % seq_id)

            # HR frame
            hr_seq_folder = os.path.join(seq_folder, 'hr')
            os.makedirs(hr_seq_folder, exist_ok=True)
            hr_filename = os.path.join(hr_seq_folder, filename_template % frames_in_seq_count)
            cv.imwrite(hr_filename, tile)

            # LR frame
            lr_seq_folder = os.path.join(seq_folder, 'lr')
            os.makedirs(lr_seq_folder, exist_ok=True)
            lr_filename = os.path.join(lr_seq_folder, filename_template % frames_in_seq_count)

            h, w, _ = tile.shape
            tile_downsampled = imresize(tile, output_shape=(h // resize_factor, w // resize_factor), method='bicubic')

            cv.imwrite(lr_filename, tile_downsampled)

        frames_in_seq_count += 1
        if frames_in_seq_count == seq_length:
            common_index += 1
            frames_in_seq_count = 0


def process_frames_video(files: list,
                         dst_root: str,
                         tile_size: int,
                         resize_factor: int,
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

            # HR frame
            hr_seq_folder = os.path.join(seq_folder, 'hr')
            os.makedirs(hr_seq_folder, exist_ok=True)
            hr_filename = os.path.join(hr_seq_folder, filename_template % frames_in_seq_count)
            cv.imwrite(hr_filename, tile)

            # LR frame
            lr_seq_folder = os.path.join(seq_folder, 'lr')
            os.makedirs(lr_seq_folder, exist_ok=True)
            lr_filename = os.path.join(lr_seq_folder, filename_template % frames_in_seq_count)

            h, w, _ = tile.shape
            tile_downsampled = imresize(tile, output_shape=(h // resize_factor, w // resize_factor), method='bicubic')

            cv.imwrite(lr_filename, tile_downsampled)

        frames_in_seq_count += 1
        if frames_in_seq_count == seq_length:
            common_index += 1
            frames_in_seq_count = 0


def generate_data(src_root: str,
                  dst_root: str,
                  tile_size: int,
                  resize_factor: int,
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
        # Check frames shapes
        shape = None
        files = sorted(glob(os.path.join(folder, '*')))
        for file in files:
            frame = cv.imread(file)
            if shape is not None:
                assert frame.shape == shape
            shape = frame.shape

        # Process frames
        out_folder = os.path.join(dst_root, dirname_template % folder_index)
        os.makedirs(out_folder, exist_ok=True)
        process_frames(files, out_folder, tile_size, resize_factor, seq_length)


def generate_data_video(src_video: str,
                        dst_root: str,
                        tile_size: int,
                        resize_factor: int,
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
    # Check frames shapes
    logger.info("Splitting frames to tiles...")
    shape = None
    files = sorted(glob(os.path.join(video_frames_path, '*')))
    for file in files:
        frame = cv.imread(file)
        if shape is not None:
            assert frame.shape == shape
        shape = frame.shape

    # Process frames
    out_folder = os.path.join(dst_root, video_filename + "_data")
    os.makedirs(out_folder, exist_ok=True)
    process_frames_video(files, out_folder, tile_size, resize_factor, seq_length)
    shutil.rmtree(video_frames_path, ignore_errors=True)


def encode_folder(src_files,
                  out_path,
                  crf: int = 0,
                  preset: str = 'ultrafast'):
    call([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', src_files,
        '-c:v', 'libx265',
        '-x265-params', 'crf=' + str(crf),
        '-preset', preset,
        '-f', 'hevc',
        '-y',
        out_path
    ])

    return out_path


def video2frames(src_path: str,
                 out_files: str):
    frame_cnt = 0
    cap_encoded = cv.VideoCapture(src_path)

    pbar = tqdm(total=get_video_length(src_path, True))
    while cap_encoded.isOpened():
        ret, frame = cap_encoded.read()
        if not ret: break
        cv.imwrite(out_files % frame_cnt, frame)
        frame_cnt += 1
        pbar.update(1)
    pbar.close()


def encode_data(root: str,
                crf_low: int = 0,
                crf_high: int = 51,
                subdir: str = "lr",
                filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    seq_folders = sorted(glob(root + "/*/*"))
    for folder in tqdm(seq_folders):
        # Encode LR frames
        src_files = os.path.join(folder, subdir, filename_template)
        out_h265_path = os.path.join(folder, subdir + ".h265")

        crf = int(np.random.uniform(crf_low, crf_high, 1))
        encode_folder(src_files, out_h265_path, crf)

        # .h265 file to frames
        lr_h265_folder = os.path.join(folder, subdir + '_h265')
        lr_h265_out = os.path.join(lr_h265_folder, filename_template)
        shutil.rmtree(lr_h265_folder, ignore_errors=True)
        os.makedirs(lr_h265_folder, exist_ok=True)
        video2frames(out_h265_path, lr_h265_out)

        # Check files
        lr_h265_length = len(sorted(glob(os.path.join(lr_h265_folder, "*"))))
        lr_length = len(sorted(glob(os.path.join(folder, subdir + "/*"))))
        # assert lr_length == lr_h265_length, f'Failed on {lr_h265_folder}'
        if lr_length != lr_h265_length:
            logger.warning(f'Failed to encode {folder}. Remove.')
            shutil.rmtree(folder, ignore_errors=True)


def decode_ftvsr(root: str,
                 chunk_size: int,
                 weights_path: str,
                 device: str = 'cuda',
                 filename_template: str = "%05d.png"):
    # Create device
    device = torch.device(device)

    # Create model
    model = FTVSR(None)
    model.to(device)
    model.eval()

    # Create checkpointer
    checkpointer = CheckPointer(model)
    checkpointer.load(weights_path)

    # Process frames
    seq_folders = sorted(glob(root + "/*/*"))
    for folder in tqdm(seq_folders):
        lr_h265_folder = os.path.join(folder, 'lr_h265')
        filelist = sorted(glob(os.path.join(lr_h265_folder, "*")))

        # Create result dir
        hr_ftvsr_folder = os.path.join(folder, 'hr_ftvsr')
        shutil.rmtree(hr_ftvsr_folder, ignore_errors=True)
        os.makedirs(hr_ftvsr_folder, exist_ok=True)

        # Process each group of frames
        cnt = 0
        while filelist:
            chunk, filelist = filelist[:chunk_size], filelist[chunk_size:]
            # Create tensors
            tensors = []
            for file in chunk:
                frame = cv.imread(file)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                tensors.append(tensor)
            tensors = torch.stack(tensors, dim=1)  # (n, t, c, h, w)

            # Infer model
            with torch.no_grad():
                outputs = model.forward(tensors.to(device))
            outputs = outputs.squeeze(dim=0)  # (t, c, h, w)

            # Tensor to frames
            ftvsr_frames = 255.0 * outputs.cpu().numpy().astype(np.float32).transpose((0, 2, 3, 1))  # (t, h, w, c)
            ftvsr_frames = np.clip(ftvsr_frames, 0, 255).astype(np.uint8)

            # Save frames
            for frame in ftvsr_frames:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                img_path = os.path.join(hr_ftvsr_folder, filename_template % cnt)
                cv.imwrite(img_path, frame)
                cnt += 1


def generate_foreground_masks(root: str,
                              prob: float = 0.8,
                              device: str = 'cuda',
                              allowed_classes = ['car', 'bus', 'train', 'truck', 'motorcycle', 'bicycle'],
                              filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Create model
    detector = YOLOv6Detector(model_type='yolov6l', device=device)

    # Process frames
    logger.info('Generating foreground masks...')

    seq_folders = sorted(glob(root + "/*/*"))
    for folder in tqdm(seq_folders):
        hr_filelist = sorted(glob(os.path.join(folder, 'hr', '*')))

        # Create result dirs
        masks_folder = os.path.join(folder, 'masks')
        shutil.rmtree(masks_folder, ignore_errors=True)
        os.makedirs(masks_folder, exist_ok=True)

        cnt = 0
        for hr_file in hr_filelist:
            hr_frame = cv.imread(hr_file)

            # Detect objects
            hr_tensor = torch.from_numpy(hr_frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            hr_tensor = hr_tensor.to(device)
            [bboxes] = detector.detect(hr_tensor)
            bboxes = bboxes.cpu().numpy()
            bboxes = detector.filter_bboxes_by_class(bboxes, allowed_classes=allowed_classes)
            bboxes = detector.filter_bboxes_by_score(bboxes, prob)

            # Create foreground mask
            h, w, c = hr_frame.shape
            mask = np.zeros(shape=(h, w, 1), dtype=np.uint8)
            for bb in bboxes:
                mask[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = 1
            mask = mask * 255

            # Save mask
            img_path = os.path.join(masks_folder, filename_template % cnt)
            cv.imwrite(img_path, mask)
            cnt += 1


def delete_useless_seq(root: str,
                       min_non_empty_masks: float,
                       min_mask_area: float):
    logger = logging.getLogger(_LOGGER_NAME)

    assert 0 <= min_mask_area <= 1.0
    assert 0 <= min_non_empty_masks <= 1.0

    logger.info('Filtering out empty sequences (without foreground objects)...')

    dirs_to_remove = []
    seq_folders = sorted(glob(root + "/*/*"))
    for folder in tqdm(seq_folders):
        masks_dir = os.path.join(folder, "masks")
        files = sorted(glob(os.path.join(masks_dir, '*')))

        cnt = 0
        for file in files:
            mask = cv.imread(file)
            white_pixels = np.count_nonzero(mask) / mask.size
            if white_pixels >= min_mask_area:
                cnt += 1

        if (not len(files)) or (cnt / len(files) < min_non_empty_masks):
            dirs_to_remove.append(folder)

    for dir in tqdm(dirs_to_remove):
        shutil.rmtree(dir, ignore_errors=True)


def generate_residuals(root: str,
                       scale_factor: int,
                       filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    logger.info("Generating residual frames...")
    seq_folders = sorted(glob(root + "/*/*"))
    for folder in tqdm(seq_folders):
        hr_filelist = sorted(glob(os.path.join(folder, 'hr', '*')))
        lr_filelist = sorted(glob(os.path.join(folder, 'lr', '*')))

        # Create result dirs
        result_folder = os.path.join(folder, 'resids')
        shutil.rmtree(result_folder, ignore_errors=True)
        os.makedirs(result_folder, exist_ok=True)

        upsample_module = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # Process frames
        cnt = 0
        for hr_file, lr_file in zip(hr_filelist, lr_filelist):
            hr_frame = cv.imread(hr_file)
            lr_frame = cv.imread(lr_file)

            lrup_frame = upsample_as_torch(lr_frame, upsample_module)
            resids = hr_frame.astype(np.float32) - lrup_frame.astype(np.float32)
            resids = (255. + resids) / 2.
            resids = resids.astype(np.uint8)

            # Save residual
            img_path = os.path.join(result_folder, filename_template % cnt)
            cv.imwrite(img_path, resids)
            cnt += 1


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Sequences Dataset Generator')
    parser.add_argument('--src-root', dest='src_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/huawei/sources/train/poly4_0/poly4_0.mkv",
                        help="Path to REDS dataset root directory or to video file")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/huawei/outputs/train/poly4_0____test",
                        help="Path where to save result dataset")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=160,
                        help="Size of output high-resolution frames")
    parser.add_argument('--resize-factor', dest='resize_factor', type=int, default=4,
                        help="Resize factor for low-resolution frames")
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=16,
                        help="Number of frames in output sequences")
    parser.add_argument('--crf-low', dest='crf_low', type=int, default=0,
                        help="Lower bound of CRF uniform distribution for HEVC encoding")
    parser.add_argument('--crf-high', dest='crf_high', type=int, default=35,
                        help="Higher bound of CRF uniform distribution for HEVC encoding")
    parser.add_argument('--video-mode', dest='video_mode', required=False, default=True, type=str2bool,
                        help="Choose video or REDS dataset frames as inputs")
    # parser.add_argument('--detect-score', dest='detect_score', type=float, default=0.25,
    #                     help="Score threshold for foreground objectes detection")
    # parser.add_argument('--non-empty-pct', dest='non_empty_pct', type=float, default=0.8,
    #                     help="Percentage of non-empty masks")
    # parser.add_argument('--min-mask-area', dest='min_mask_area', type=float, default=0.25,
    #                     help="Percentage of non-empty area")
    args = parser.parse_args()

    # Create logger
    logger = setup_logger(_LOGGER_NAME, dist_util.get_rank())
    logger.info(args)

    # Generate tile frames from set of frames or video
    if args.video_mode:
        generate_data_video(args.src_root, args.dst_root, args.tile_size, args.resize_factor, args.seq_length)
    else:
        generate_data(args.src_root, args.dst_root, args.tile_size, args.resize_factor, args.seq_length)

    # Encode low-resolution frames by HEVC
    assert 0 <= args.crf_low <= 51
    assert args.crf_low <= args.crf_high <= 51
    encode_data(args.dst_root, args.crf_low, args.crf_high, 'lr')

    # Step 2 - Create foreground masks
    # allowed_classes = ['car', 'bus', 'train', 'truck', 'motorcycle', 'bicycle', 'person']
    # generate_foreground_masks(args.dst_root, prob=args.detect_score, allowed_classes=allowed_classes)

    # Step 3 (optional) - Remove sequences without foreground objects
    # delete_useless_seq(args.dst_root, args.non_empty_pct, args.min_mask_area)

    # Step 4 - Generate residual frames 
    # generate_residuals(args.dst_root, args.resize_factor)

    # Step 5 - Encode frames by HEVC
    # encode_data(args.dst_root, 25, 35, 'lr')
    # encode_data(args.dst_root, 25, 35, 'resids')


if __name__ == '__main__':
    main()

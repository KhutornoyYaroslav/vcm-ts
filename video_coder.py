import os
import torch
import shutil
import pickle
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from tqdm import tqdm
from subprocess import call
from facenet_pytorch import MTCNN
from core.utils import dist_util
from core.config import codec_settings
from core.utils.logger import setup_logger, logging
from core.config.default import _CFG as cfg
from core.utils.video import get_video_length, get_video_resolution
from core.utils.checkpoint import CheckPointer
from core.utils.matlab_imresize import imresize
from core.utils.tiling import frame_to_tiles, tiles_to_frame
from core.modelling.model import _MODEL_ARCHITECTURES
from core.modelling.model.lpdetector import LiplateDetectionPredictor


_LOGGER_NAME = "CODEC"
_PATHS_ARTIFACTS_HR = "artifacts/hr"
_PATHS_ARTIFACTS_LR = "artifacts/lr"
_PATHS_ARTIFACTS_LR_ENCODED = "artifacts/lr_h265"
_PATHS_ARTIFACTS_CUTOUT = "artifacts/cutout"
_PATHS_ARTIFACTS_CUTOUT_ENCODED = "artifacts/cutout_h265"
_PATHS_ARTIFACTS_VSR = "artifacts/vsr_frames"
_PATHS_ARTIFACTS_RESULT = "artifacts/result_frames"
_PATHS_ENCODED_DIR = "encoded"
_PATHS_DECODED_DIR = "decoded"
_PATHS_INFO = "info"


def video_to_frames(video_path: str,
                    result_root: str,
                    subdir: str,
                    filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Open video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file '{video_path}'")
        return 0
    
    # Create result directory
    res_folder = os.path.join(result_root, subdir)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Process video
    logger.info("Splitting video to frames")
    cnt = 0
    pbar = tqdm(total=get_video_length(video_path, True))
    while 1:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imwrite(os.path.join(res_folder, filename_template % cnt), frame)
        cnt += 1
        pbar.update(1)
    pbar.close()
    logger.info(f"Video splitting results in {cnt} frames")

    return cnt


def downsample_and_save_video(video_path: str,
                              result_root: str,
                              downsample_factor: int = 4,
                              filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Open video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file '{video_path}'")
        return 0
    
    # Create result directory
    res_folder = os.path.join(result_root, _PATHS_ARTIFACTS_LR)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Downsample and save frames
    logger.info(f"Downsampling x{downsample_factor} frames")
    cnt = 0
    pbar = tqdm(total=get_video_length(video_path, True))
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        frame_ds = imresize(frame, 1.0 / downsample_factor, 'bicubic')
        cv.imwrite(os.path.join(res_folder, filename_template % cnt), frame_ds)
        cnt += 1
        pbar.update(1)
    pbar.close()
    logger.info(f"Video downsampling results in {cnt} frames")

    return cnt


def downsample_and_save_frames(result_root: str,
                               downsample_factor: int = 4,
                               filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan frames
    hr_folder = os.path.join(result_root, _PATHS_ARTIFACTS_HR)
    hr_filelist = sorted(glob(os.path.join(hr_folder, "*")))
    
    # Create result directory
    res_folder = os.path.join(result_root, _PATHS_ARTIFACTS_LR)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Downsample and save frames
    logger.info(f"Downsampling x{downsample_factor} frames")
    cnt = 0
    pbar = tqdm(total=len(hr_filelist))
    for hr_file in hr_filelist:
        hr_frame = cv.imread(hr_file)
        frame_ds = imresize(hr_frame, 1.0 / downsample_factor, 'bicubic')
        cv.imwrite(os.path.join(res_folder, filename_template % cnt), frame_ds)
        cnt += 1
        pbar.update(1)
    pbar.close()
    logger.info(f"Video downsampling results in {cnt} frames")

    return cnt


def encode_folder(src_files, out_path, crf: int, preset: str = 'ultrafast', pix_fmt: str = 'gbrp'):
    call([ 
        'ffmpeg',
        '-i', src_files,
        '-pix_fmt', pix_fmt,
        '-c:v', 'libx265',
        '-preset', preset,
        '-crf', str(crf),
        '-y', out_path
    ])

    return out_path


def encode_frames(src_root: str,
                  video_path: str,
                  crf: int,
                  preset = 'medium',
                  pix_fmt: str = 'gbrp',
                  save_to_frames = True,
                  frames_path: str = '',
                  filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan frames
    src_files = os.path.join(src_root, filename_template)

    # Call encoder
    logger.info(f"Encoding '{src_files}' frames to '{video_path}'")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    encode_folder(src_files, video_path, crf=crf, preset=preset, pix_fmt=pix_fmt)

    # Encoded video to frames
    if save_to_frames:
        shutil.rmtree(frames_path, ignore_errors=True)
        os.makedirs(frames_path, exist_ok=True)
        video_to_frames(video_path, frames_path, '', filename_template)

        # Check lengths
        src_length = len(sorted(glob(src_root)))
        dst_length = len(sorted(glob(frames_path)))
        assert src_length == dst_length


def do_vsr(result_root: str,
           model_arch: str,
           model_num_blocks: int,
           model_mid_channels: int,
           chunk_size: int,
           tile_size: int,
           tile_padding: int,
           weights_path: str,
           device: str = 'cuda',
           filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Create device
    device = torch.device(device)

    # Create model
    logger.info(f"Creating '{model_arch}' model")
    cfg.MODEL.NUM_BLOCKS = model_num_blocks
    cfg.MODEL.MID_CHANNELS = model_mid_channels
    model = _MODEL_ARCHITECTURES[model_arch](cfg)
    model.to(device)
    model.eval()

    # Load model weights
    logger.info(f"Initializing model parameters by '{weights_path}'")
    checkpointer = CheckPointer(model)
    checkpointer.load(weights_path)

    # Create result dir
    res_folder = os.path.join(result_root, _PATHS_ARTIFACTS_VSR)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Process frames
    lr_folder = os.path.join(result_root, _PATHS_ARTIFACTS_LR_ENCODED)
    lr_filelist = sorted(glob(os.path.join(lr_folder, "*")))

    # Process each group of frames
    logger.info(f"Decoding super resolution frames")
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
            lr_tiles = frame_to_tiles(lr_frame, tile_size, tile_padding)
            lr_chunk_tiles.append(lr_tiles)
        lr_chunk_tiles = np.array(lr_chunk_tiles) # (T, N, H, W, C)

        # Process each group of tiles
        vsr_chunk_tiles = []
        for i in range(lr_chunk_tiles.shape[1]):
            # LR tensor
            lr_tiles = lr_chunk_tiles[:, i] # (T, H, W, C)
            lr_tensor = torch.from_numpy(lr_tiles.astype(np.float32) / 255.0).permute(0, 3, 1, 2) # (T, C, H, W)
            lr_tensor = lr_tensor.unsqueeze(0) # (B, T, C, H, W)

            # Infer model
            with torch.no_grad():
                outputs = model.forward(lr_tensor.to(device))
            outputs = outputs.squeeze(dim=0) # (T, C, H, W)
            outputs = torch.clamp(outputs, 0., 1.)

            # Process model output
            vsr_tiles = outputs.permute(0, 2, 3, 1)
            vsr_tiles = 255 * vsr_tiles.cpu().detach().numpy()
            vsr_tiles = vsr_tiles.astype(np.uint8)
            vsr_chunk_tiles.append(vsr_tiles)
        vsr_chunk_tiles = np.array(vsr_chunk_tiles) # (N, T, 4*H, 4*W, C)

        # Save frames
        for i in range(vsr_chunk_tiles.shape[1]):
            vsr_tiles = vsr_chunk_tiles[:, i]
            vsr_frame = tiles_to_frame(vsr_tiles, 4 * h_orig, 4 * w_orig, 4 * tile_padding)
            vsr_frame = cv.cvtColor(vsr_frame, cv.COLOR_RGB2BGR)
            img_path = os.path.join(res_folder, filename_template % cnt)
            cv.imwrite(img_path, vsr_frame)
            cnt += 1
            pbar.update(1)
    pbar.close()
    logger.info(f"Decoded {cnt} super resolution frames")


def detect_liplates(root: str,
                    prob: float = 0.8,
                    padding: int = 0,
                    device: str = 'cuda',
                    filename_template: str = "%05d"):
    logger = logging.getLogger(_LOGGER_NAME)

    assert prob > 0.0
    assert padding >= 0

    # Create device
    device = torch.device(device)

    # Scan frames
    src_folder = os.path.join(root, _PATHS_ARTIFACTS_HR)
    filelist = sorted(glob(os.path.join(src_folder, "*")))

    # Create result dir
    res_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Create liplates detector
    lp_detector = LiplateDetectionPredictor()
    lp_detector.setup(weights='Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/best.pt',
                    device=device)

    # Process frames
    logger.info('Detecting license plates')
    cnt = 0
    pbar = tqdm(total=len(filelist))
    for src_file in filelist:
        src_frame = cv.imread(src_file)
        src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
        h, w, c = src_frame.shape

        # Detect liplates
        [lp_preds] = lp_detector.predict(src_frame)
        lp_preds = lp_preds.cpu().numpy()

        # Filter liplates
        lp_coords = []
        for pred in lp_preds:
            score = pred[4]
            if score < prob: continue
            x1, y1, x2, y2 = pred[:4]
            x1 = max(min(int(x1 - padding), w), 0)
            x2 = max(min(int(x2 + padding), w), 0)
            y1 = max(min(int(y1 - padding), h), 0)
            y2 = max(min(int(y2 + padding), h), 0)
            lp_coords.append([x1, y1, x2, y2])

        # Serialize coordinates
        filename = os.path.join(res_folder, filename_template % cnt)
        file = open(filename, 'wb')
        pickle.dump(np.array(lp_coords, dtype=np.uint16), file)
        file.close()
        cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f"License plates coordinates saved to '{res_folder}'")


def detect_faces(root: str,
                 prob: float = 0.8,
                 padding: int = 0,
                 device: str = 'cuda',
                 filename_template: str = "%05d"):
    logger = logging.getLogger(_LOGGER_NAME)

    assert prob > 0.0
    assert padding >= 0

    # Create device
    device = torch.device(device)

    # Scan frames
    src_folder = os.path.join(root, _PATHS_ARTIFACTS_HR)
    filelist = sorted(glob(os.path.join(src_folder, "*")))

    # Create result dir
    res_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Create faces detector
    face_detector = MTCNN(device=device)
    face_detector.eval()

    # Process frames
    logger.info('Detecting faces')
    cnt = 0
    pbar = tqdm(total=len(filelist))
    for src_file in filelist:
        src_frame = cv.imread(src_file)
        src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
        h, w, c = src_frame.shape

        # Detect faces
        face_bbs, face_probs = face_detector.detect(src_frame, landmarks=False)

        # Filter faces
        face_coords = []
        if face_bbs is not None:
            for bbox, score in zip(face_bbs, face_probs):
                if score < prob: continue
                x1, y1, x2, y2 = bbox[0:4]
                x1 = max(min(int(x1 - padding), w), 0)
                x2 = max(min(int(x2 + padding), w), 0)
                y1 = max(min(int(y1 - padding), h), 0)
                y2 = max(min(int(y2 + padding), h), 0)
                face_coords.append([x1, y1, x2, y2])

        # Serialize coordinates
        filename = os.path.join(res_folder, filename_template % cnt)
        file = open(filename, 'wb')
        pickle.dump(np.array(face_coords, dtype=np.uint16), file)
        file.close()
        cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f"Faces coordinates saved to '{res_folder}'")


def create_gradient_mask(w, h, border_size: int):
    if border_size > 0:
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for i, x in enumerate(np.linspace(0.9, 0.0, border_size)):
            mask[i:h-i, i:w-i, :] = (1 - x)
    else:
        mask = np.ones(shape=(h, w, 1), dtype=np.float32)

    return mask


def fuse_layers(root: str,
                faces_enable: bool = True,
                liplates_enable: bool = True,
                faces_padding: int = 0,
                liplates_padding: int = 0,
                filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan files
    vsr_folder = os.path.join(root, _PATHS_ARTIFACTS_VSR)
    vsr_filelist = sorted(glob(os.path.join(vsr_folder, "*")))

    cutout_folder = os.path.join(root, _PATHS_ARTIFACTS_CUTOUT_ENCODED)
    cutout_filelist = sorted(glob(os.path.join(cutout_folder, "*")))

    if liplates_enable:
        liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
        liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))
        assert len(liplates_coords_filelist) == len(vsr_filelist)

    if faces_enable:
        faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
        faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))
        assert len(faces_coords_filelist) == len(vsr_filelist)

    # Create result dirs
    res_folder = os.path.join(root, _PATHS_ARTIFACTS_RESULT)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Process frames
    logger.info('Creating result frames')
    cnt = 0
    pbar = tqdm(total=len(vsr_filelist))
    for vsr_file, cutout_file in zip(vsr_filelist, cutout_filelist):
        # Read frames
        vsr_frame = cv.imread(vsr_file)
        cutout_frame = cv.imread(cutout_file)

        # Read liplates bounding boxes
        lp_bboxes = []
        if liplates_enable:
            f = open(liplates_coords_filelist[cnt], 'rb')
            lp_bboxes = pickle.load(f)
            f.close()

        # Read faces bounding boxes
        face_bboxes = []
        if faces_enable:
            f = open(faces_coords_filelist[cnt], 'rb')
            face_bboxes = pickle.load(f)
            f.close()

        # Create mask
        h, w, c = vsr_frame.shape
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1:y2, x1:x2] = create_gradient_mask(w=x2-x1, h=y2-y1, border_size=liplates_padding)
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1:y2, x1:x2] = create_gradient_mask(w=x2-x1, h=y2-y1, border_size=faces_padding)

        # Process
        result_frame = (1 - mask) * vsr_frame.astype(np.float32)
        result_frame += mask * cutout_frame.astype(np.float32)
        result_frame = np.clip(result_frame, 0, 255)
        result_frame = result_frame.astype(np.uint8)

        # Save result
        img_path = os.path.join(res_folder, filename_template % cnt)
        cv.imwrite(img_path, result_frame)
        cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f'Created {cnt} result frames')


def create_cutout_layer_frames(root: str,
                               faces_enable: bool = True,
                               liplates_enable: bool = True,
                               filename_template: str = "%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan files
    hr_folder = os.path.join(root, _PATHS_ARTIFACTS_HR)
    hr_filelist = sorted(glob(os.path.join(hr_folder, "*")))

    if liplates_enable:
        liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
        liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))
        assert len(liplates_coords_filelist) == len(hr_filelist)

    if faces_enable:
        faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
        faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))
        assert len(faces_coords_filelist) == len(hr_filelist)

    # Create result dir
    res_folder = os.path.join(root, _PATHS_ARTIFACTS_CUTOUT)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Process frames
    logger.info('Creating cutout layer frames...')
    save_cnt = 0
    pbar = tqdm(total=len(hr_filelist))
    for file_idx, hr_file in enumerate(hr_filelist):
        # Read frame
        hr_frame = cv.imread(hr_file)

        # Read liplates bounding boxes
        lp_bboxes = []
        if liplates_enable:
            f = open(liplates_coords_filelist[file_idx], 'rb')
            lp_bboxes = pickle.load(f)
            f.close()

        # Read faces bounding boxes
        face_bboxes = []
        if faces_enable:
            f = open(faces_coords_filelist[file_idx], 'rb')
            face_bboxes = pickle.load(f)
            f.close()

        # Create foreground mask
        h, w, c = hr_frame.shape
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1:y2, x1:x2] = 1.0
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1:y2, x1:x2] = 1.0

        # Create result frame
        result_frame = hr_frame.astype(np.float32) * mask
        result_frame = result_frame.astype(np.uint8)

        # Save mask
        img_path = os.path.join(res_folder, filename_template % save_cnt)
        cv.imwrite(img_path, result_frame)
        save_cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f"Cutout layer frames saved to '{res_folder}'")


def get_dir_size(start_path: str = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def calc_bitrate_metrics(root: str,
                         video_path: str):
    logger = logging.getLogger(_LOGGER_NAME)

    logger.info('Calculating bitrate metrics...')

    frames_num = get_video_length(video_path)
    frame_w, frame_h = get_video_resolution(video_path)
    total_pixels = frames_num * frame_w * frame_h
    assert total_pixels > 0

    src_size = 8 * os.path.getsize(video_path)
    # encoded_size = 8 * get_dir_size(start_path=os.path.join(root, _PATHS_ENCODED_DIR))
    video_layer_size = 8 * os.path.getsize(os.path.join(root, _PATHS_ENCODED_DIR, 'video_layer.h265'))
    cutout_layer_size = 8 * os.path.getsize(os.path.join(root, _PATHS_ENCODED_DIR, 'cutout_layer.h265'))
    encoded_size = video_layer_size + cutout_layer_size # In the real case, the bitrate of liplates/faces coordinates is negligible

    src_bpp = src_size / total_pixels
    encoded_bpp = encoded_size / total_pixels
    video_layer_bpp = video_layer_size / total_pixels
    cutout_layer_bpp = cutout_layer_size / total_pixels

    metrics_file = os.path.join(root, _PATHS_INFO, 'bitrate_metrics.txt')
    logger.info(f'Saving bitrate metrics to {metrics_file}')

    with open(metrics_file, 'w') as f:
        f.write(f'Results for: {video_path}\n')
        f.write(f'Source kB: {src_size / 8 / 1024}\n')
        f.write(f'Encoded kB: {encoded_size / 8 / 1024}\n')
        f.write(f'Source bpp: {src_bpp}\n')
        f.write(f'Encoded bpp [total]: {encoded_bpp}\n')
        f.write(f'Encoded bpp [video layer]: {video_layer_bpp}\n')
        f.write(f'Encoded bpp [cutout layer]: {cutout_layer_bpp}\n')
        f.write(f'Recompression ratio [total]: {src_bpp / encoded_bpp}\n')
        f.write(f'Recompression ratio [video layer]: {src_bpp / video_layer_bpp}\n')
        f.write(f'Recompression ratio [cutout layer]: {src_bpp / cutout_layer_bpp}\n')


def calc_visual_metrics(root: str,
                        video_path: str,
                        liplates_padding: int = 0,
                        faces_padding: int = 0):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan files
    hr_folder = os.path.join(root, _PATHS_ARTIFACTS_HR)
    hr_filelist = sorted(glob(os.path.join(hr_folder, "*")))

    result_folder = os.path.join(root, _PATHS_ARTIFACTS_RESULT)
    result_filelist = sorted(glob(os.path.join(result_folder, "*")))

    liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
    liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))

    faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
    faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))

    # Process frames
    logger.info('Calculating PSNR metrics...')

    psnrs, psnrs_vsr, psnrs_cutout = [], [], []
    pbar = tqdm(total=len(hr_filelist))
    for file_idx, _ in enumerate(hr_filelist):
        # Read frames
        hr_frame = cv.imread(hr_filelist[file_idx])
        result_frame = cv.imread(result_filelist[file_idx])

        # Read liplates bounding boxes
        lp_bboxes = []
        if len(liplates_coords_filelist) == len(hr_filelist):
            f = open(liplates_coords_filelist[file_idx], 'rb')
            lp_bboxes = pickle.load(f)
            f.close()

        # Read faces bounding boxes
        face_bboxes = []
        if len(faces_coords_filelist) == len(hr_filelist):
            f = open(faces_coords_filelist[file_idx], 'rb')
            face_bboxes = pickle.load(f)
            f.close()

        # Create cutout layer mask
        h, w, c = hr_frame.shape
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1+liplates_padding:y2-liplates_padding,
                 x1+liplates_padding:x2-liplates_padding] = 1.0
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1+faces_padding:y2-faces_padding,
                 x1+faces_padding:x2-faces_padding] = 1.0
            
        # Calculate PSNR
        mask_nonzeros = np.count_nonzero(mask)
        mask_zeros = hr_frame.size - mask_nonzeros

        mse = (hr_frame.astype(np.float32) / 255.0 - result_frame.astype(np.float32) / 255.0)**2
        mse_vsr = mse * (1.0 - mask)
        mse_cutout = mse * mask

        psnr = 10 * np.log10(1.0 / np.mean(mse))
        psnr_vsr = 10 * np.log10(1.0 / (np.sum(mse_vsr) / mask_zeros))
        psnr_cutout = 10 * np.log10(1.0 / (np.sum(mse_cutout) / mask_nonzeros))

        psnrs.append(psnr)
        psnrs_vsr.append(psnr_vsr)
        psnrs_cutout.append(psnr_cutout)

        pbar.update(1)
    pbar.close()

    metrics_file = os.path.join(root, _PATHS_INFO, 'psnr_metrics.txt')
    logger.info(f'Saving PSNR metrics to {metrics_file}')

    with open(metrics_file, 'w') as f:
        f.write(f'Results for: {video_path}\n')
        f.write(f'Total PSNR [RGB format]: {np.mean(psnrs)}\n')
        f.write(f'VSR PSNR [RGB format]: {np.mean(psnrs_vsr)}\n')
        f.write(f'Cutout layer PSNR [RGB format]: {np.mean(psnrs_cutout)}\n')


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='SVS Video Codec')
    parser.add_argument('--video-path', dest='video_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/datasets/reds_val_003",
                        help="Path to video to recompress")
    parser.add_argument("--settings-file", dest="settings_file", type=str, default="configs/codec_settings.yaml", metavar="FILE",
                        help="Path to codec settings file")
    parser.add_argument('--result-root', dest='result_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/datasets/reds_val_003",
                        help="Path where to save results")
    parser.add_argument('--do-encode', dest='do_encode', required=False, type=str2bool, default=True,
                        help="Encoding enable")
    parser.add_argument('--do-decode', dest='do_decode', required=False, type=str2bool, default=True,
                        help="Decoding enable")
    args = parser.parse_args()

    # Create logger
    logger = setup_logger(_LOGGER_NAME, dist_util.get_rank())
    logger.info(args)

    # Create settings
    codec_settings.merge_from_file(args.settings_file)
    codec_settings.freeze()

    # Create settings dump
    os.makedirs(os.path.join(args.result_root, _PATHS_INFO), exist_ok=True)
    with open(os.path.join(args.result_root, _PATHS_INFO, 'codec_settings.yaml'), "w") as cfg_dump:
        cfg_dump.write(str(codec_settings))

    # Encode
    if args.do_encode:
        # ---------------- ENCODER [VIDEO LAYER] ----------------
        # video_to_frames(video_path=args.video_path,
        #                 result_root=args.result_root,
        #                 subdir=_PATHS_ARTIFACTS_HR)

        downsample_and_save_frames(result_root=args.result_root,
                                downsample_factor=4)
        
        encode_frames(src_root=os.path.join(args.result_root, _PATHS_ARTIFACTS_LR),
                    video_path=os.path.join(args.result_root, _PATHS_ENCODED_DIR, 'video_layer.h265'),
                    crf=codec_settings.VIDEO_LAYER.H265.CRF,
                    preset=codec_settings.VIDEO_LAYER.H265.PRESET,
                    pix_fmt=codec_settings.VIDEO_LAYER.H265.PIX_FMT,
                    save_to_frames=True,
                    frames_path=os.path.join(args.result_root, _PATHS_ARTIFACTS_LR_ENCODED))
        
        # ---------------- ENCODER [CUTOUT LAYER] ----------------
        if codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.ENABLE:
            detect_liplates(root=args.result_root,
                            prob=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.PROB,
                            padding=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.PADDING,
                            device=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.DEVICE)
            
        if codec_settings.CUTOUT_LAYER.DETECTORS.FACES.ENABLE:
            detect_faces(root=args.result_root,
                        prob=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.PROB,
                        padding=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.PADDING,
                        device=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.DEVICE)

        create_cutout_layer_frames(root=args.result_root,
                                faces_enable=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.ENABLE,
                                liplates_enable=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.ENABLE)

        encode_frames(src_root=os.path.join(args.result_root, _PATHS_ARTIFACTS_CUTOUT),
                    video_path=os.path.join(args.result_root, _PATHS_ENCODED_DIR, 'cutout_layer.h265'),
                    crf=codec_settings.CUTOUT_LAYER.H265.CRF,
                    preset=codec_settings.CUTOUT_LAYER.H265.PRESET,
                    pix_fmt=codec_settings.CUTOUT_LAYER.H265.PIX_FMT,
                    save_to_frames=True,
                    frames_path=os.path.join(args.result_root, _PATHS_ARTIFACTS_CUTOUT_ENCODED))
        
        # calc_bitrate_metrics(root=args.result_root,
        #                      video_path=args.video_path)

    # Decode
    if args.do_decode:
        # ---------------- DECODER ----------------
        do_vsr(result_root=args.result_root,
            model_arch=codec_settings.VIDEO_LAYER.VSR_MODEL.ARCHITECTURE,
            model_num_blocks=codec_settings.VIDEO_LAYER.VSR_MODEL.NUM_BLOCKS,
            model_mid_channels=codec_settings.VIDEO_LAYER.VSR_MODEL.MID_CHANNELS,
            chunk_size=codec_settings.VIDEO_LAYER.VSR_MODEL.CHUNK_SIZE,
            tile_size=codec_settings.VIDEO_LAYER.VSR_MODEL.TILE_SIZE,
            tile_padding=codec_settings.VIDEO_LAYER.VSR_MODEL.TILE_PADDING,
            weights_path=codec_settings.VIDEO_LAYER.VSR_MODEL.WEIGHTS_PATH,
            device=codec_settings.VIDEO_LAYER.VSR_MODEL.DEVICE)

        fuse_layers(root=args.result_root,
                    faces_enable=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.ENABLE,
                    liplates_enable=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.ENABLE,
                    faces_padding=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.PADDING,
                    liplates_padding=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.PADDING)

        encode_frames(src_root=os.path.join(args.result_root, _PATHS_ARTIFACTS_RESULT),
                    video_path=os.path.join(args.result_root, _PATHS_DECODED_DIR, 'svs.h265'),
                    crf=0,
                    preset='medium',
                    pix_fmt='gbrp',
                    save_to_frames=False)
        
        # calc_visual_metrics(root=args.result_root,
        #                     video_path=args.video_path,
        #                     liplates_padding=codec_settings.CUTOUT_LAYER.DETECTORS.LIPLATES.PADDING,
        #                     faces_padding=codec_settings.CUTOUT_LAYER.DETECTORS.FACES.PADDING)


if __name__ == '__main__':
    main()

import argparse
import json
import os
import pickle
import shutil
import time
from glob import glob
from subprocess import call

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from tqdm import tqdm
from ultralytics import YOLO

from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.models.video_model import DMC
from DCVC_HEM.src.utils.common import interpolate_log
from DCVC_HEM.src.utils.png_reader import PNGReader
from DCVC_HEM.src.utils.stream_helper import get_state_dict, get_padding_size, np_image_to_tensor, save_torch_image
from core.config import codec_settings
from core.utils import dist_util
from core.utils.logger import setup_logger, logging
from core.utils.video import get_video_length, get_video_resolution

_LOGGER_NAME = "CODEC"
_PATHS_ARTIFACTS_SOURCE_FRAMES = "artifacts/source_frames"
_PATHS_ARTIFACTS_DCVC_HEM = "artifacts/dcvc_hem"
_PATHS_ARTIFACTS_RESIDUALS = "artifacts/residuals"
_PATHS_ARTIFACTS_RESIDUALS_ENCODED = "artifacts/residuals_h265"
_PATHS_ARTIFACTS_RESULT = "artifacts/result_frames"
_PATHS_ARTIFACTS_SAME_BITRATE = "artifacts/same_bitrate"
_PATHS_ENCODED_DIR = "encoded"
_PATHS_DECODED_DIR = "decoded"
_PATHS_INFO = "info"


def video_to_frames(video_path: str,
                    result_root: str,
                    subdir: str,
                    filename_template: str = "im%05d.png"):
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
    decode_time = 0
    pbar = tqdm(total=get_video_length(video_path, True))
    while 1:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        cv.imwrite(os.path.join(res_folder, filename_template % (cnt + 1)), frame)
        t1 = time.time()
        cnt += 1
        decode_time += t1 - t0
        pbar.update(1)
    pbar.close()
    logger.info(f"H.265 decoding time {(decode_time) / cnt * 1000:.2f} ms for {cnt} frames")
    logger.info(f"Video splitting results in {cnt} frames")

    return cnt


def run_dcvc(video_net, i_frame_net, args, device):
    logger = logging.getLogger(_LOGGER_NAME)
    frame_num = args['frame_num']
    gop = args['gop']
    write_stream = 'write_stream' in args and args['write_stream']
    src_reader = PNGReader(args['img_path'])

    frame_types = []
    bits = []
    frame_pixel_num = 0

    decoded_frames_folder = args['decoded_frame_folder']
    shutil.rmtree(decoded_frames_folder, ignore_errors=True)
    os.makedirs(decoded_frames_folder, exist_ok=True)
    encoding_time = 0
    decoding_time = 0

    with torch.no_grad():
        for frame_idx in tqdm(range(frame_num)):
            rgb, png_path = src_reader.read_one_frame(src_format="rgb", get_png_path=True)
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # Pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args['bin_folder'],
                                    f"im{str(frame_idx + 1).zfill(5)}.bin") if write_stream else None

            if frame_idx % gop == 0:
                result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                result = video_net.encode_decode(x_padded, dpb, bin_path,
                                                 pic_height=pic_height, pic_width=pic_width,
                                                 mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                 y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

            encoding_time += result['encoding_time']
            decoding_time += result['decoding_time']

            save_path = os.path.join(decoded_frames_folder, f'im{str(frame_idx + 1).zfill(5)}.png')
            save_torch_image(x_hat, save_path)

    logger.info(f"Average encoding time {encoding_time / frame_num * 1000:.2f} ms")
    logger.info(f"Average decoding time {decoding_time / frame_num * 1000:.2f} ms")
    logger.info(f"Reserved GPU memory {torch.cuda.memory_reserved() / 1024 / 1024:.2f} mb")


def encode_decode_dcvc(frames_dir: str,
                       image_model_weights: str,
                       video_model_weights: str,
                       anchor_num: int,
                       gop: int,
                       rate_count: int,
                       quality: int,
                       write_stream: bool,
                       device: str,
                       out_frames_dir: str,
                       out_bins_dir: str):
    assert image_model_weights != "", "Invalid image model weights"
    assert video_model_weights != "", "Invalid video model weights"
    logger = logging.getLogger(_LOGGER_NAME)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # Set q_scale for image model (quality)
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(image_model_weights)
    max_q_scale = i_frame_q_scales[0]
    min_q_scale = i_frame_q_scales[-1]
    i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)
    i_frame_q_scale = i_frame_q_scales[quality]

    # Set q_scale for video model (quality)
    p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(video_model_weights)
    max_y_q_scale = p_frame_y_q_scales[0]
    min_y_q_scale = p_frame_y_q_scales[-1]
    p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_count)
    p_frame_y_q_scale = p_frame_y_q_scales[quality]
    max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
    min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
    p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_count)
    p_frame_mv_y_q_scale = p_frame_mv_y_q_scales[quality]

    # Initialize image and video models
    i_state_dict = get_state_dict(image_model_weights)
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict, strict=False)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    p_state_dict = get_state_dict(video_model_weights)
    video_net = DMC(anchor_num=anchor_num)
    video_net.load_state_dict(p_state_dict, strict=False)
    video_net = video_net.to(device)
    video_net.eval()

    if write_stream:
        video_net.update(force=True)
        i_frame_net.update(force=True)

    # Run encode and decode with DCVC-HEM
    shutil.rmtree(out_frames_dir, ignore_errors=True)
    os.makedirs(out_frames_dir, exist_ok=True)
    shutil.rmtree(out_bins_dir, ignore_errors=True)
    os.makedirs(out_bins_dir, exist_ok=True)
    frame_num = len(glob(os.path.join(frames_dir, "*.png")))
    args = dict(
        i_frame_q_scale=i_frame_q_scale,
        p_frame_y_q_scale=p_frame_y_q_scale,
        p_frame_mv_y_q_scale=p_frame_mv_y_q_scale,
        gop=gop,
        frame_num=frame_num,
        write_stream=write_stream,
        bin_folder=out_bins_dir,
        img_path=frames_dir,
        decoded_frame_folder=out_frames_dir
    )
    logger.info('Encoding/decoding with DCVC-HEM')
    run_dcvc(video_net, i_frame_net, args, device)


def detect_liplates(root: str,
                    prob: float = 0.8,
                    padding: int = 0,
                    device: str = 'cuda',
                    filename_template: str = "%05d"):
    logger = logging.getLogger(_LOGGER_NAME)
    torch.cuda.empty_cache()

    assert prob > 0.0
    assert padding >= 0

    # Create device
    device = torch.device(device)

    # Scan frames
    src_folder = os.path.join(root, _PATHS_ARTIFACTS_SOURCE_FRAMES)
    source_images = sorted(glob(os.path.join(src_folder, "*.png")))

    # Create result dir
    res_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Create liplates detector
    lp_detector = YOLO('pretrained/yolov8-lp.pt')
    lp_detector = lp_detector.to(device)

    # Process frames
    logger.info('Detecting license plates')
    src_reader = PNGReader(src_folder)
    detect_time = 0
    for i in tqdm(range(len(source_images))):
        rgb = src_reader.read_one_frame(src_format="rgb")
        image = np_image_to_tensor(rgb)
        image = image.to(device)
        t0 = time.time()
        n, c, h, w = image.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, p=32)
        image_padded = torch.nn.functional.pad(
            image,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )

        # Detect liplates
        lp_preds = lp_detector(image_padded, imgsz=(image_padded.shape[2], image_padded.shape[3]), verbose=False)[0]
        boxes = lp_preds.boxes.xyxy.cpu()
        scores = lp_preds.boxes.conf.cpu()

        # Filter liplates
        lp_coords = []
        for box, score in zip(boxes, scores):
            if score < prob: continue
            x1, y1, x2, y2 = box
            x1 = max(min(int(x1 - padding), w), 0)
            x2 = max(min(int(x2 + padding), w), 0)
            y1 = max(min(int(y1 - padding), h), 0)
            y2 = max(min(int(y2 + padding), h), 0)
            lp_coords.append([x1, y1, x2, y2])
        t1 = time.time()
        detect_time += t1 - t0

        # Serialize coordinates
        filename = os.path.join(res_folder, filename_template % (i + 1))
        file = open(filename, 'wb')
        pickle.dump(np.array(lp_coords, dtype=np.uint16), file)
        file.close()
    logger.info(f"Average detecting license plates time {detect_time / len(source_images) * 1000:.2f} ms")
    logger.info(f"Reserved GPU memory {torch.cuda.memory_reserved() / 1024 / 1024:.2f} mb")
    logger.info(f"License plates coordinates saved to '{res_folder}'")


def detect_faces(root: str,
                 prob: float = 0.8,
                 padding: int = 0,
                 device: str = 'cuda',
                 filename_template: str = "%05d"):
    logger = logging.getLogger(_LOGGER_NAME)
    torch.cuda.empty_cache()

    assert prob > 0.0
    assert padding >= 0

    # Create device
    device = torch.device(device)

    # Scan frames
    src_folder = os.path.join(root, _PATHS_ARTIFACTS_SOURCE_FRAMES)
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
    cnt = 1
    detect_time = 0
    pbar = tqdm(total=len(filelist))
    for src_file in filelist:
        src_frame = cv.imread(src_file)
        src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
        h, w, c = src_frame.shape

        # Detect faces
        t0 = time.time()
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
        t1 = time.time()
        detect_time += t1 - t0

        # Serialize coordinates
        filename = os.path.join(res_folder, filename_template % cnt)
        file = open(filename, 'wb')
        pickle.dump(np.array(face_coords, dtype=np.uint16), file)
        file.close()
        cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f"Average detecting faces time {detect_time / len(filelist) * 1000:.2f} ms")
    logger.info(f"Reserved GPU memory {torch.cuda.memory_reserved() / 1024 / 1024:.2f} mb")
    logger.info(f"Faces coordinates saved to '{res_folder}'")


def compute_residuals(root: str,
                      use_liplates: bool,
                      use_faces: bool,
                      out_residuals_dir: str,
                      filename_template: str = "im%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan frames
    source_frames = sorted(glob(os.path.join(root, _PATHS_ARTIFACTS_SOURCE_FRAMES, "*.png")))
    encoded_frames = sorted(glob(os.path.join(root, _PATHS_ARTIFACTS_DCVC_HEM, "*.png")))

    if use_liplates:
        liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
        liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))
        assert len(liplates_coords_filelist) == len(source_frames)

    if use_faces:
        faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
        faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))
        assert len(faces_coords_filelist) == len(source_frames)

    # Create result dir
    shutil.rmtree(out_residuals_dir, ignore_errors=True)
    os.makedirs(out_residuals_dir, exist_ok=True)

    # Process frames
    logger.info('Compute residuals')
    cnt = 1
    residual_time = 0
    mask_time = 0
    pbar = tqdm(total=len(source_frames))
    for source_frame_path, encoded_frame_path in zip(source_frames, encoded_frames):
        # Compute
        source_frame = cv.imread(source_frame_path).astype(np.float32)
        encoded_frame = cv.imread(encoded_frame_path).astype(np.float32)
        h, w, c = source_frame.shape
        t0_residual = time.time()
        residual = source_frame - encoded_frame
        residual = np.clip(residual + 128, 0.0, 255.0)
        t1_residual = time.time()
        residual_time += t1_residual - t0_residual

        # Read liplates bounding boxes
        lp_bboxes = []
        if use_liplates:
            f = open(liplates_coords_filelist[cnt - 1], 'rb')
            lp_bboxes = pickle.load(f)
            f.close()

        # Read faces bounding boxes
        face_bboxes = []
        if use_faces:
            f = open(faces_coords_filelist[cnt - 1], 'rb')
            face_bboxes = pickle.load(f)
            f.close()

        # Create foreground mask
        t0_mask = time.time()
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1:y2, x1:x2] = 1.0
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1:y2, x1:x2] = 1.0

        # Create result frame
        result_frame = residual.astype(np.float32) * mask
        result_frame = result_frame.astype(np.uint8)
        t1_mask = time.time()
        mask_time += t1_mask - t0_mask

        # Save result
        img_path = os.path.join(out_residuals_dir, filename_template % cnt)
        cv.imwrite(img_path, result_frame)
        cnt += 1
        pbar.update(1)
    pbar.close()
    logger.info(f"Average residual computing time {residual_time / len(source_frames) * 1000:.2f} ms")
    logger.info(f"Average masking time {mask_time / len(source_frames) * 1000:.2f} ms")
    logger.info(f"Residuals saved to '{out_residuals_dir}'")


def encode_folder_crf(src_files, out_path, crf: int, preset: str = 'ultrafast', pix_fmt: str = 'gbrp'):
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


def encode_folder_bitrate(src_files, out_path, bitrate: int, preset: str = 'ultrafast', pix_fmt: str = 'gbrp'):
    call([
        'ffmpeg',
        '-i', src_files,
        '-pix_fmt', pix_fmt,
        '-c:v', 'libx265',
        '-preset', preset,
        '-b:v', str(bitrate) + 'k',
        '-y', out_path
    ])

    return out_path


def encode_frames(src_root: str,
                  video_path: str,
                  crf: int,
                  preset='medium',
                  pix_fmt: str = 'gbrp',
                  save_to_frames=True,
                  frames_path: str = '',
                  filename_template: str = "im%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan frames
    src_files = os.path.join(src_root, filename_template)

    # Call encoder
    logger.info(f"Encoding '{src_files}' frames to '{video_path}'")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    t0 = time.time()
    encode_folder_crf(src_files, video_path, crf=crf, preset=preset, pix_fmt=pix_fmt)
    t1 = time.time()
    source_frames = sorted(glob(os.path.join(src_root, "*.png")))
    logger.info(f"H.265 encoding time {(t1 - t0) / len(source_frames) * 1000:.2f} ms for {len(source_frames)} frames")

    # Encoded video to frames
    if save_to_frames:
        shutil.rmtree(frames_path, ignore_errors=True)
        os.makedirs(frames_path, exist_ok=True)
        video_to_frames(video_path, frames_path, '', filename_template)

        # Check lengths
        src_length = len(sorted(glob(src_root)))
        dst_length = len(sorted(glob(frames_path)))
        assert src_length == dst_length


def create_gradient_mask(w, h, border_size: int):
    if border_size > 0:
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for i, x in enumerate(np.linspace(0.9, 0.0, border_size)):
            mask[i:h - i, i:w - i, :] = (1 - x)
    else:
        mask = np.ones(shape=(h, w, 1), dtype=np.float32)

    return mask


def fuse_layers(root: str,
                faces_enable: bool = True,
                liplates_enable: bool = True,
                faces_padding: int = 0,
                liplates_padding: int = 0,
                filename_template: str = "im%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan files
    dcvc_hem_folder = os.path.join(root, _PATHS_ARTIFACTS_DCVC_HEM)
    dcvc_hem_filelist = sorted(glob(os.path.join(dcvc_hem_folder, "*.png")))

    enhancement_folder = os.path.join(root, _PATHS_ARTIFACTS_RESIDUALS)
    enhancement_filelist = sorted(glob(os.path.join(enhancement_folder, "*.png")))

    if liplates_enable:
        liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
        liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))
        assert len(liplates_coords_filelist) == len(dcvc_hem_filelist)

    if faces_enable:
        faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
        faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))
        assert len(faces_coords_filelist) == len(dcvc_hem_filelist)

    # Create result dirs
    res_folder = os.path.join(root, _PATHS_ARTIFACTS_RESULT)
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)

    # Process frames
    logger.info('Creating result frames')
    cnt = 0
    fuse_time = 0
    pbar = tqdm(total=len(dcvc_hem_filelist))
    for dcvc_hem_file, enhancement_file in zip(dcvc_hem_filelist, enhancement_filelist):
        # Read frames
        dcvc_hem_frame = cv.imread(dcvc_hem_file).astype(np.float32)
        enhancement_frame_residual = cv.imread(enhancement_file).astype(np.float32)
        enhancement_frame = enhancement_frame_residual - 128

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
        h, w, c = dcvc_hem_frame.shape
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1:y2, x1:x2] = create_gradient_mask(w=x2 - x1, h=y2 - y1, border_size=liplates_padding)
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1:y2, x1:x2] = create_gradient_mask(w=x2 - x1, h=y2 - y1, border_size=faces_padding)

        # Process
        t0 = time.time()
        result_frame = dcvc_hem_frame
        result_frame += mask * enhancement_frame
        result_frame = np.clip(result_frame, 0, 255)
        result_frame = result_frame.astype(np.uint8)
        t1 = time.time()
        fuse_time += t1 - t0

        # Save result
        img_path = os.path.join(res_folder, filename_template % (cnt + 1))
        cv.imwrite(img_path, result_frame)
        cnt += 1

        pbar.update(1)
    pbar.close()
    logger.info(f"Average fusing time {fuse_time / cnt * 1000:.2f} ms")
    logger.info(f'Created {cnt} result frames')


def encode_same_bitrate(root: str,
                        source_video_path: str,
                        out_video_path: str,
                        preset='medium',
                        pix_fmt: str = 'gbrp',
                        save_to_frames=True,
                        frames_path: str = '',
                        filename_template: str = "im%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    logger.info('Encode H.265 with same bitrate as encoded video...')

    # Get source video duration
    cap = cv.VideoCapture(source_video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Calculate bitrate
    enhancement_layer_size = 8 * os.path.getsize(os.path.join(root, _PATHS_ENCODED_DIR, 'enhancement_layer.h265'))
    base_layer_size = 8 * get_dir_size(os.path.join(root, _PATHS_ENCODED_DIR, 'dcvc_hem_bins'))
    encoded_size = enhancement_layer_size + base_layer_size
    bitrate = int(encoded_size / duration / 1000)  # kBit

    # Call encoder
    src_files = os.path.join(root, _PATHS_ARTIFACTS_RESULT, filename_template)
    encode_folder_bitrate(src_files, out_video_path, bitrate=bitrate, preset=preset, pix_fmt=pix_fmt)

    # Encoded video with same bitrate to frames
    if save_to_frames:
        shutil.rmtree(frames_path, ignore_errors=True)
        os.makedirs(frames_path, exist_ok=True)
        video_to_frames(out_video_path, frames_path, '', filename_template)

        # Check lengths
        src_length = len(sorted(glob(os.path.join(root, _PATHS_ARTIFACTS_RESULT))))
        dst_length = len(sorted(glob(frames_path)))
        assert src_length == dst_length


def get_dir_size(start_path: str = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def calc_bitrate_metrics(root: str,
                         video_path: str,
                         gop: int):
    logger = logging.getLogger(_LOGGER_NAME)

    logger.info('Calculating bitrate metrics...')

    frames_num = get_video_length(video_path)
    frame_w, frame_h = get_video_resolution(video_path)
    total_pixels = frames_num * frame_w * frame_h
    assert total_pixels > 0

    src_size = 8 * os.path.getsize(video_path)
    enhancement_layer_size = 8 * os.path.getsize(os.path.join(root, _PATHS_ENCODED_DIR, 'enhancement_layer.h265'))
    base_layer_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'dcvc_hem_bins')
    base_layer_size = 8 * get_dir_size(base_layer_folder)
    encoded_size = enhancement_layer_size + base_layer_size

    src_bpp = src_size / total_pixels
    encoded_bpp = encoded_size / total_pixels
    enhancement_layer_bpp = enhancement_layer_size / total_pixels
    base_layer_bpp = base_layer_size / total_pixels

    metrics_file = os.path.join(root, _PATHS_INFO, 'bitrate_metrics.txt')
    logger.info(f'Saving bitrate metrics to {metrics_file}')

    with open(metrics_file, 'w') as f:
        f.write(f'Results for: {video_path}\n')
        f.write(f'Source kB: {src_size / 8 / 1024}\n')
        f.write(f'Encoded kB: {encoded_size / 8 / 1024}\n')
        f.write(f'Source bpp: {src_bpp}\n')
        f.write(f'Encoded bpp [total]: {encoded_bpp}\n')
        f.write(f'Encoded bpp [enhancement layer]: {enhancement_layer_bpp}\n')
        f.write(f'Encoded bpp [base layer]: {base_layer_bpp}\n')
        f.write(f'Recompression ratio [total]: {src_bpp / encoded_bpp}\n')
        f.write(f'Recompression ratio [enhancement layer]: {src_bpp / enhancement_layer_bpp}\n')
        f.write(f'Recompression ratio [base layer]: {src_bpp / base_layer_bpp}\n')

    log_result = {
        'gop': gop,
        'avg_bpp': encoded_bpp
    }
    json_path = os.path.join(root, _PATHS_INFO, 'quality.json')
    with open(json_path, 'w') as fp:
        json.dump(log_result, fp)


def calc_visual_metrics(root: str,
                        video_path: str,
                        liplates_padding: int = 0,
                        faces_padding: int = 0):
    logger = logging.getLogger(_LOGGER_NAME)

    # Scan files
    source_folder = os.path.join(root, _PATHS_ARTIFACTS_SOURCE_FRAMES)
    source_filelist = sorted(glob(os.path.join(source_folder, "*.png")))

    result_folder = os.path.join(root, _PATHS_ARTIFACTS_RESULT)
    result_filelist = sorted(glob(os.path.join(result_folder, "*.png")))

    same_bitrate_folder = os.path.join(root, _PATHS_ARTIFACTS_SAME_BITRATE)
    same_bitrate_filelist = sorted(glob(os.path.join(same_bitrate_folder, "*.png")))

    liplates_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'liplates_coords')
    liplates_coords_filelist = sorted(glob(os.path.join(liplates_coords_folder, "*")))

    faces_coords_folder = os.path.join(root, _PATHS_ENCODED_DIR, 'faces_coords')
    faces_coords_filelist = sorted(glob(os.path.join(faces_coords_folder, "*")))

    # Process frames
    logger.info('Calculating PSNR metrics...')

    psnrs, psnrs_dcvc_hem, psnrs_enhancement, psnrs_same_bitrate = [], [], [], []
    pbar = tqdm(total=len(source_filelist))
    for file_idx, _ in enumerate(source_filelist):
        # Read frames
        hr_frame = cv.imread(source_filelist[file_idx])
        result_frame = cv.imread(result_filelist[file_idx])
        same_bitrate_frame = cv.imread(same_bitrate_filelist[file_idx])

        # Read liplates bounding boxes
        lp_bboxes = []
        if len(liplates_coords_filelist) == len(source_filelist):
            f = open(liplates_coords_filelist[file_idx], 'rb')
            lp_bboxes = pickle.load(f)
            f.close()

        # Read faces bounding boxes
        face_bboxes = []
        if len(faces_coords_filelist) == len(source_filelist):
            f = open(faces_coords_filelist[file_idx], 'rb')
            face_bboxes = pickle.load(f)
            f.close()

        # Create cutout layer mask
        h, w, c = hr_frame.shape
        mask = np.zeros(shape=(h, w, 1), dtype=np.float32)
        for [x1, y1, x2, y2] in lp_bboxes:
            mask[y1 + liplates_padding:y2 - liplates_padding,
            x1 + liplates_padding:x2 - liplates_padding] = 1.0
        for [x1, y1, x2, y2] in face_bboxes:
            mask[y1 + faces_padding:y2 - faces_padding,
            x1 + faces_padding:x2 - faces_padding] = 1.0

        # Calculate PSNR
        mask_nonzeros = np.count_nonzero(mask)
        mask_zeros = hr_frame.size - mask_nonzeros

        mse = (hr_frame.astype(np.float32) / 255.0 - result_frame.astype(np.float32) / 255.0) ** 2
        mse_dcvc_hem = mse * (1.0 - mask)
        mse_enhancement = mse * mask
        mse_same_bitrate = (hr_frame.astype(np.float32) / 255.0 - same_bitrate_frame.astype(np.float32) / 255.0) ** 2

        psnr = 10 * np.log10(1.0 / np.mean(mse))
        psnr_dcvc_hem = 10 * np.log10(1.0 / (np.sum(mse_dcvc_hem) / mask_zeros))
        psnr_enhancement = 10 * np.log10(1.0 / (np.sum(mse_enhancement) / mask_nonzeros))
        psnr_same_bitrate = 10 * np.log10(1.0 / np.mean(mse_same_bitrate))

        psnrs.append(psnr)
        psnrs_dcvc_hem.append(psnr_dcvc_hem)
        psnrs_enhancement.append(psnr_enhancement)
        psnrs_same_bitrate.append(psnr_same_bitrate)

        pbar.update(1)
    pbar.close()

    metrics_file = os.path.join(root, _PATHS_INFO, 'psnr_metrics.txt')
    logger.info(f'Saving PSNR metrics to {metrics_file}')

    with open(metrics_file, 'w') as f:
        f.write(f'Results for: {video_path}\n')
        f.write(f'Total PSNR [RGB format]: {np.mean(psnrs)}\n')
        f.write(f'DCVC-HEM PSNR [RGB format]: {np.mean(psnrs_dcvc_hem)}\n')
        f.write(f'Enhancement layer PSNR [RGB format]: {np.mean(psnrs_enhancement)}\n')
        f.write(f'H265 encoded with same bitrate as total PSNR [RGB format]: {np.mean(psnrs_same_bitrate)}\n')


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Video Coding for Machines for Traffic Scenes')
    parser.add_argument('--video-path', dest='video_path', type=str,
                        default="data/huawei/outputs/benchmark/test_21_short.mp4",
                        help="Path to video to recompress")
    parser.add_argument("--settings-file", dest="settings_file", type=str, default="configs/codec_settings.yaml",
                        metavar="FILE",
                        help="Path to codec settings file")
    parser.add_argument('--result-root', dest='result_root', type=str,
                        default="outputs/codec/test_21_short",
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
        # ---------------- COMMON PART ----------------
        video_to_frames(video_path=args.video_path,
                        result_root=args.result_root,
                        subdir=_PATHS_ARTIFACTS_SOURCE_FRAMES)

        # ---------------- ENCODER [BASE LAYER] ----------------
        encode_decode_dcvc(frames_dir=os.path.join(args.result_root, _PATHS_ARTIFACTS_SOURCE_FRAMES),
                           image_model_weights=codec_settings.BASE_LAYER.DCVC_HEM.I_FRAME_WEIGHTS,
                           video_model_weights=codec_settings.BASE_LAYER.DCVC_HEM.P_FRAME_WEIGHTS,
                           anchor_num=codec_settings.BASE_LAYER.DCVC_HEM.ANCHOR_NUM,
                           gop=codec_settings.BASE_LAYER.DCVC_HEM.GOP,
                           rate_count=codec_settings.BASE_LAYER.DCVC_HEM.RATE_COUNT,
                           quality=codec_settings.BASE_LAYER.DCVC_HEM.QUALITY,
                           write_stream=codec_settings.BASE_LAYER.DCVC_HEM.WRITE_STREAM,
                           device=codec_settings.BASE_LAYER.DCVC_HEM.DEVICE,
                           out_frames_dir=os.path.join(args.result_root, _PATHS_ARTIFACTS_DCVC_HEM),
                           out_bins_dir=os.path.join(args.result_root, _PATHS_ENCODED_DIR, 'dcvc_hem_bins'))

        # ---------------- ENCODER [ENHANCEMENT LAYER] ----------------
        if codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.ENABLE:
            detect_liplates(root=args.result_root,
                            prob=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PROB,
                            padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PADDING,
                            device=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.DEVICE)

        if codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.ENABLE:
            detect_faces(root=args.result_root,
                         prob=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.PROB,
                         padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.PADDING,
                         device=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.DEVICE)

        compute_residuals(root=args.result_root,
                          use_liplates=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.ENABLE,
                          use_faces=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.ENABLE,
                          out_residuals_dir=os.path.join(args.result_root, _PATHS_ARTIFACTS_RESIDUALS))

        encode_frames(src_root=os.path.join(args.result_root, _PATHS_ARTIFACTS_RESIDUALS),
                      video_path=os.path.join(args.result_root, _PATHS_ENCODED_DIR, 'enhancement_layer.h265'),
                      crf=codec_settings.ENHANCEMENT_LAYER.H265.CRF,
                      preset=codec_settings.ENHANCEMENT_LAYER.H265.PRESET,
                      pix_fmt=codec_settings.ENHANCEMENT_LAYER.H265.PIX_FMT,
                      save_to_frames=True,
                      frames_path=os.path.join(args.result_root, _PATHS_ARTIFACTS_RESIDUALS_ENCODED))

        calc_bitrate_metrics(root=args.result_root,
                             video_path=args.video_path,
                             gop=codec_settings.BASE_LAYER.DCVC_HEM.GOP)

    # Decode
    if args.do_decode:
        # ---------------- DECODER ----------------
        fuse_layers(root=args.result_root,
                    faces_enable=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.ENABLE,
                    liplates_enable=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.ENABLE,
                    faces_padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.PADDING,
                    liplates_padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PADDING)

        encode_frames(src_root=os.path.join(args.result_root, _PATHS_ARTIFACTS_RESULT),
                      video_path=os.path.join(args.result_root, _PATHS_DECODED_DIR, 'vcm-ts_decoded.h265'),
                      crf=0,
                      preset='medium',
                      pix_fmt='gbrp',
                      save_to_frames=False)

        encode_same_bitrate(root=args.result_root,
                            source_video_path=args.video_path,
                            out_video_path=os.path.join(args.result_root, _PATHS_DECODED_DIR, 'same_bitrate.h265'),
                            preset=codec_settings.COMPARE.H265.PRESET,
                            pix_fmt=codec_settings.COMPARE.H265.PIX_FMT,
                            save_to_frames=True,
                            frames_path=os.path.join(args.result_root, _PATHS_ARTIFACTS_SAME_BITRATE))

        calc_visual_metrics(root=args.result_root,
                            video_path=args.video_path,
                            liplates_padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PADDING,
                            faces_padding=codec_settings.ENHANCEMENT_LAYER.DETECTORS.FACES.PADDING)


if __name__ == '__main__':
    main()

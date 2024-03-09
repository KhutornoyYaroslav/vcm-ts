# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
from glob import glob
from subprocess import call

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.models.video_model import DMC
from DCVC_HEM.src.utils.common import interpolate_log
from DCVC_HEM.src.utils.png_reader import PNGReader
from DCVC_HEM.src.utils.stream_helper import get_padding_size, get_state_dict


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop = args['gop']
    src_reader = PNGReader(args['img_path'])

    bits = []
    frame_pixel_num = 0

    temp_dir = os.path.join(args['decoded_frame_folder'], 'temp')
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    with torch.no_grad():
        for frame_idx in tqdm(range(frame_num)):
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            if frame_idx % gop == 0:
                result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'],
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                bits.append(result["bit"])
            else:
                result = video_net.encode_decode(x_padded, dpb,
                                                 pic_height=pic_height, pic_width=pic_width,
                                                 mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                 y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                bits.append(result['bit'])

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

            save_path = os.path.join(temp_dir, f'im{str(frame_idx + 1).zfill(5)}.png')
            save_torch_image(x_hat, save_path)

    bpp = sum(bits) / (frame_num * frame_pixel_num)
    frames_dir = os.path.join(args['decoded_frame_folder'], str(bpp) + "_" + str(gop))
    os.rename(temp_dir, frames_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def worker(device, i_frame_net, video_net, args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)

    run_test(video_net, i_frame_net, args, device)


def decod_dcvc(dataset_dir: str,
               gop: int,
               rate_count: int,
               out_dir: str,
               model_name: str,
               config):
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = 1
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    # Задаётся количество используемых при тестировании значений q_scale (min = 2, max = 64)
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(config['image_model_weights'])
    if len(i_frame_q_scales) == rate_count:
        pass
    else:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)

    # Аналогично извлекаются из весов для видео значения y_q_scale (резидуалы) и mv_q_scale (motion vector)
    p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(config['video_model_weights'])
    if len(p_frame_y_q_scales) == rate_count:
        pass
    else:
        max_y_q_scale = p_frame_y_q_scales[0]
        min_y_q_scale = p_frame_y_q_scales[-1]
        p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_count)

        max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
        min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
        p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_count)

    device = config['device']

    # Загрузка весов модели для изображений и её дальнейшая инициализация
    i_state_dict = get_state_dict(config['image_model_weights'])
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict, strict=False)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    # Загрузка весов модели для видео и её дальнейшая инициализация
    p_state_dict = get_state_dict(config['video_model_weights'])
    video_net = DMC()
    video_net.load_state_dict(p_state_dict, strict=False)
    video_net = video_net.to(device)
    video_net.eval()

    # Подготовка аргументов для DCVC-HEM
    video_folders = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    model_dir = os.path.join(out_dir, model_name)
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    for video_folder in video_folders:
        obj_seq = []
        images_path = os.path.join(video_folder.path, "images")
        frame_num = len(glob(os.path.join(images_path, "*.png")))
        decoded_frame_folder = os.path.join(model_dir, video_folder.name)
        shutil.rmtree(decoded_frame_folder, ignore_errors=True)
        os.makedirs(decoded_frame_folder, exist_ok=True)
        for rate_idx in range(rate_count):
            cur_args = dict(
                rate_idx=rate_idx,
                i_frame_q_scale=i_frame_q_scales[rate_idx],
                p_frame_y_q_scale=p_frame_y_q_scales[rate_idx],
                p_frame_mv_y_q_scale=p_frame_mv_y_q_scales[rate_idx],
                gop=gop,
                frame_num=frame_num,
                img_path=images_path,
                decoded_frame_folder=decoded_frame_folder
            )

            obj = threadpool_executor.submit(
                worker,
                device,
                i_frame_net,
                video_net,
                cur_args)
            obj_seq.append(obj)
        objs.append(obj_seq)

    results = []
    for seq_index, video_folder in zip(range(len(objs)), video_folders):
        print(f'Video: {video_folder.name}')
        results_seq = []
        for index, obj in enumerate(objs[seq_index]):
            print(f'\tRate: {index + 1}')
            result = obj.result()
            results_seq.append(result)
        results.append(results_seq)


def get_video_bpp(path, countable=True):
    cap = cv2.VideoCapture(path)
    original_video_size = os.path.getsize(path) * 8
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if countable:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        count = 0
        while cap.isOpened():
            ret, x = cap.read()
            if not ret:
                break
            count += 1

    return original_video_size / count / w / h


def video_to_frames(video_path: str, out_dir: str, gop: int):
    os.makedirs(out_dir, exist_ok=True)

    bpp = get_video_bpp(video_path, countable=False)
    frames_dir = os.path.join(out_dir, str(bpp) + "_" + str(gop))
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=int(frames_count))
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_filename = os.path.join(frames_dir, 'im' + str(i).zfill(5) + '.png')
        cv2.imwrite(image_filename, frame)
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()


def encode_folder(src_files, out_path, framerate: int, crf: int = 0, preset: str = 'ultrafast'):
    call([
        'ffmpeg',
        '-hide_banner',
        '-framerate', str(framerate),
        '-loglevel', 'error',
        '-i', src_files,
        '-c:v', 'libx265',
        '-x265-params', 'crf=' + str(crf),
        '-preset', preset,
        '-f', 'hevc',
        '-y',
        out_path
    ])


def decod_hevc(dataset_dir: str,
               out_dir: str,
               rate_num: int,
               crf_start: int,
               crf_end: int,
               fps: int,
               gop: int):
    crfs = np.linspace(crf_start, crf_end, num=rate_num, dtype=np.int32).tolist()
    model_dir = os.path.join(out_dir, 'hevc')
    temp_dir = os.path.join(model_dir, 'temp')

    video_folders = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    for video_folder in video_folders:
        frames_dir = os.path.join(video_folder.path, 'images', "im%05d.png")
        result_dir = os.path.join(model_dir, video_folder.name)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)

        for crf in crfs:
            out_filename_crf_custom = os.path.join(temp_dir, "crf_" + str(crf) + ".mp4")
            encode_folder(frames_dir, out_filename_crf_custom, framerate=fps, crf=crf)
            video_to_frames(out_filename_crf_custom, result_dir, gop)

        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Benchmark models decoding')
    parser.add_argument('--config', dest='config', type=str,
                        default="benchmark_config_decoding.json",
                        help="Config for benchmark")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")  # for DCVC-HEM executor

    with open(args.config) as f:
        config = json.load(f)

    dataset_dir = config['dataset_dir']
    gop = config['gop']
    rate_count = config['rate_count']
    out_dir = config['out_dir']
    for key, value in config['models'].items():
        print(f'Decoding with {key}')
        if key == 'HEVC':
            crf_start = config['models'][key]['crf_start']
            crf_end = config['models'][key]['crf_end']
            fps = config['models'][key]['fps']
            decod_hevc(dataset_dir, out_dir, rate_count, crf_start, crf_end, fps, gop)
        elif 'DCVC-HEM' in key:
            decod_dcvc(dataset_dir, gop, rate_count, out_dir, key, value)
        else:
            raise AttributeError("Invalid model in config file")


if __name__ == "__main__":
    main()

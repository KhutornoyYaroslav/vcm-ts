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
from tqdm import tqdm

from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.models.video_model import DMC
from DCVC_HEM.src.utils.common import interpolate_log
from DCVC_HEM.src.utils.png_reader import PNGReader
from DCVC_HEM.src.utils.stream_helper import get_padding_size, get_state_dict, filesize, np_image_to_tensor, \
    save_torch_image


def generate_log_json(frame_num, gop, frame_types, bits, frame_pixel_num):
    cur_ave_i_frame_bit = 0
    cur_ave_p_frame_bit = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_ave_i_frame_bit += bits[idx]
            cur_i_frame_num += 1
        else:
            cur_ave_p_frame_bit += bits[idx]
            cur_p_frame_num += 1

    log_result = {}
    log_result['gop'] = gop
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['avg_i_frame_bpp'] = cur_ave_i_frame_bit / cur_i_frame_num / frame_pixel_num
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['avg_p_frame_bpp'] = cur_ave_p_frame_bit / total_p_pixel_num
    else:
        log_result['avg_p_frame_bpp'] = 0
    log_result['avg_bpp'] = (cur_ave_i_frame_bit + cur_ave_p_frame_bit) / \
                            (frame_num * frame_pixel_num)
    log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
    log_result['frame_type'] = frame_types

    return log_result


def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop = args['gop']
    src_reader = PNGReader(args['img_path'])

    frame_types = []
    bits = []
    frame_pixel_num = 0

    temp_dir = os.path.join(args['decoded_frame_folder'], 'temp')
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

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

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            if frame_idx % gop == 0:
                if i_frame_net is not None:
                    result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'],
                                                       pic_height=pic_height, pic_width=pic_width)
                else:
                    result = {
                        "x_hat": x_padded,
                        "bit": filesize(png_path) * 8
                    }

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
                result = video_net.encode_decode(x_padded, dpb,
                                                 pic_height=pic_height, pic_width=pic_width,
                                                 mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                 y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

            save_path = os.path.join(temp_dir, f'im{str(frame_idx + 1).zfill(5)}.png')
            save_torch_image(x_hat, save_path)

    log_result = generate_log_json(frame_num, gop, frame_types, bits, frame_pixel_num)
    result_name = "quality_" + str(args['rate_idx'])
    frames_dir = os.path.join(args['decoded_frame_folder'], result_name)
    json_path = os.path.join(args['decoded_frame_folder'], result_name + '.json')
    shutil.rmtree(json_path, ignore_errors=True)
    with open(json_path, 'w') as fp:
        json.dump(log_result, fp)
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
               config):
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = 1
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    # Задаётся количество используемых при тестировании значений q_scale (min = 2, max = 64)
    if config['image_model_weights'] == "":
        i_frame_q_scales = [0] * rate_count
    else:
        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(config['image_model_weights'])
        if config['i_frame_q_scales']:
            if config['interpolate_q_scales']:
                max_q_scale = config['i_frame_q_scales'][0]
                min_q_scale = config['i_frame_q_scales'][-1]
                i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)
            else:
                assert len(config['i_frame_q_scales']) == rate_count
                i_frame_q_scales = config['i_frame_q_scales']
        elif len(i_frame_q_scales) == rate_count:
            pass
        else:
            max_q_scale = i_frame_q_scales[0]
            min_q_scale = i_frame_q_scales[-1]
            i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)

    # Аналогично извлекаются из весов для видео значения y_q_scale (резидуалы) и mv_q_scale (motion vector)
    p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(config['video_model_weights'])
    if config['p_frame_y_q_scales']:
        if config['interpolate_q_scales']:
            max_y_q_scale = config['p_frame_y_q_scales'][0]
            min_y_q_scale = config['p_frame_y_q_scales'][-1]
            p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_count)

            max_mv_y_q_scale = config['p_frame_mv_q_scales'][0]
            min_mv_y_q_scale = config['p_frame_mv_q_scales'][-1]
            p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_count)
        else:
            assert len(config['p_frame_y_q_scales']) == rate_count
            assert len(config['p_frame_mv_q_scales']) == rate_count
            p_frame_y_q_scales = config['p_frame_y_q_scales']
            p_frame_mv_y_q_scales = config['p_frame_mv_q_scales']
    elif len(p_frame_y_q_scales) == rate_count:
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
    if config['image_model_weights']:
        i_state_dict = get_state_dict(config['image_model_weights'])
        i_frame_net = IntraNoAR()
        i_frame_net.load_state_dict(i_state_dict, strict=False)
        i_frame_net = i_frame_net.to(device)
        i_frame_net.eval()
    else:
        i_frame_net = None

    # Загрузка весов модели для видео и её дальнейшая инициализация
    p_state_dict = get_state_dict(config['video_model_weights'])
    video_net = DMC(anchor_num=int(config['anchor_num']))
    video_net.load_state_dict(p_state_dict, strict=False)
    video_net = video_net.to(device)
    video_net.eval()

    # Подготовка аргументов для DCVC-HEM
    video_folders = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    model_dir = os.path.join(out_dir, config['name'])
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


def video_to_frames(video_path: str, out_dir: str, gop: int, quality_index: int):
    os.makedirs(out_dir, exist_ok=True)

    result_name = "quality_" + str(quality_index)
    json_path = os.path.join(out_dir, result_name + '.json')
    shutil.rmtree(json_path, ignore_errors=True)
    bpp = get_video_bpp(video_path, countable=False)
    log_result = dict(
        gop=gop,
        avg_bpp=bpp
    )
    with open(json_path, 'w') as fp:
        json.dump(log_result, fp)

    frames_dir = os.path.join(out_dir, result_name)
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


def encode_folder(src_files, out_path, framerate: int, crf: int = 0, gop: int = 32, preset: str = 'ultrafast'):
    call([
        'ffmpeg',
        '-hide_banner',
        '-pix_fmt', 'yuv420p',
        '-framerate', str(framerate),
        '-loglevel', 'error',
        '-i', src_files,
        '-c:v', 'libx265',
        '-x265-params', 'crf=' + str(crf) + ':keyint=' + str(gop),
        '-preset', preset,
        '-tune', 'zerolatency',
        '-f', 'hevc',
        '-y',
        out_path
    ])


def decod_hevc(dataset_dir: str,
               out_dir: str,
               rate_num: int,
               gop: int,
               config):
    crfs = np.linspace(config['crf_start'], config['crf_end'], num=rate_num, dtype=np.int32).tolist()
    codec_dir = os.path.join(out_dir, config['name'])
    temp_dir = os.path.join(codec_dir, 'temp')

    video_folders = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    shutil.rmtree(codec_dir, ignore_errors=True)
    os.makedirs(codec_dir, exist_ok=True)
    for video_folder in video_folders:
        frames_dir = os.path.join(video_folder.path, 'images', "im%05d.png")
        result_dir = os.path.join(codec_dir, video_folder.name)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)

        for index, crf in enumerate(crfs):
            out_filename_crf_custom = os.path.join(temp_dir, "crf_" + str(crf) + ".mp4")
            encode_folder(frames_dir, out_filename_crf_custom, framerate=config['fps'], crf=crf, gop=gop,
                          preset=config['preset'])
            video_to_frames(out_filename_crf_custom, result_dir, gop, index)

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
    for key, value in config['codecs'].items():
        if key == 'HEVC':
            for hevc_config in value:
                name = hevc_config['name']
                print(f'Decoding with {name}')
                decod_hevc(dataset_dir, out_dir, rate_count, gop, hevc_config)
        elif key == 'DCVC-HEM':
            for dcvc_config in value:
                name = dcvc_config['name']
                print(f'Decoding with {name}')
                decod_dcvc(dataset_dir, gop, rate_count, out_dir, dcvc_config)
        else:
            raise AttributeError("Invalid model in config file")


if __name__ == "__main__":
    main()

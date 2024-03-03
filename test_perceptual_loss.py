# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from DCVC_HEM.src.models.video_model import DMC
from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from DCVC_HEM.src.utils.stream_helper import get_padding_size, get_state_dict
from DCVC_HEM.src.utils.png_reader import PNGReader
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--i_frame_q_scales', type=float, nargs="+")
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--model_path',  type=str)
    parser.add_argument('--p_frame_y_q_scales', type=float, nargs="+")
    parser.add_argument('--p_frame_mv_y_q_scales', type=float, nargs="+")
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    # Инициализация читателя изображений
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['img_path'], args['src_width'], args['src_height'])

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    decoded = []
    frame_pixel_num = 0

    start_time = time.time()
    p_frame_number = 0

    with torch.no_grad():  # no_grad() - не перезаписывает веса
        for frame_idx in tqdm(range(frame_num)):
            # Считывание кадра из видео и перенос его на GPU
            frame_start_time = time.time()
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # Добавление паддингов на изображение (по дефолту используется 64 пикселя)
            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            # Кодирование и декодирование кадра при помощи i_frame_net, когда номер кадра в GOP равен 0 (первый),
            # или при помощи video_net
            if frame_idx % gop_size == 0:
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
                p_frame_number += 1

            # Удаление паддингов и подсчёт метрик
            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()
            psnrs.append(psnr)
            msssims.append(msssim)
            decoded.append(x_hat.cpu())
            frame_end_time = time.time()

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,",
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1]:.4f}, MS-SSIM: {msssims[-1]:.4f} ")

            if save_decoded_frame:
                save_path = os.path.join(args['decoded_frame_folder'], f'{frame_idx}.png')
                save_torch_image(x_hat, save_path)

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"encoding/decoding {p_frame_number} P frames")

    log_result = generate_log_json(frame_num, frame_types, bits, psnrs, msssims,
                                   frame_pixel_num, test_time, decoded)
    return log_result


def encode_one(args, i_frame_net, video_net, device):
    # Подготовка моделей к записи битстрима в бинарные файлы на диск (инициализация арифметических энкодеров)
    if args['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    # Подготовка аргументов для нейронки
    sub_dir_name = args['video_path']
    gop_size = args['gop']

    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name,
                                            str(args['rate_idx']))
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None

    args['img_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['gop_size'] = gop_size
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(video_net, i_frame_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(device, i_frame_net, video_net, args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)

    result = encode_one(args, i_frame_net, video_net, device)
    return result


def get_original_bboxes(rcnn, args, device):
    frame_num = args['frame_num']

    # Инициализация читателя изображений
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['img_path'], args['src_width'], args['src_height'])

    bboxes = []

    with torch.no_grad():  # no_grad() - не перезаписывает веса
        for _ in tqdm(range(frame_num)):
            # Считывание кадра из видео и перенос его на GPU
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)

            output = forward_rcnn(rcnn, x)
            del x
            torch.cuda.empty_cache()
            bboxes.append(output)

    return bboxes


def forward_rcnn(rcnn, x):
    x_cuda = x.to('cuda')  # TODO: Fix
    output = rcnn(x_cuda)[0]  # batch = 1
    del x_cuda
    torch.cuda.empty_cache()
    output['boxes'] = output['boxes'].cpu()
    output['labels'] = output['labels'].cpu()
    output['scores'] = output['scores'].cpu()
    return output


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    count_frames = 0
    count_sequences = 0

    # Задаётся количество используемых при тестировании значений q_scale (min = 2, max = 64)
    rate_num = args.rate_num
    # Извлекает из весов используемые значения q_scale (в весах доступны q_basic и q_scale) и на основе заданных
    # аргументов либо использует значения из весов, либо заданные вручную, либо интерполирует N значений в заданном
    # промежутке (минимальное и максимальное значения берутся из весов)
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    if args.i_frame_q_scales is not None:
        assert len(args.i_frame_q_scales) == rate_num
        i_frame_q_scales = args.i_frame_q_scales
    elif len(i_frame_q_scales) == rate_num:
        pass
    else:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num)

    # Аналогично извлекаются из весов для видео значения y_q_scale (резидуалы) и mv_q_scale (motion vector)
    p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(args.model_path)
    if args.p_frame_y_q_scales is not None:
        assert len(args.p_frame_y_q_scales) == rate_num
        assert len(args.p_frame_mv_y_q_scales) == rate_num
        p_frame_y_q_scales = args.p_frame_y_q_scales
        p_frame_mv_y_q_scales = args.p_frame_mv_y_q_scales
    elif len(p_frame_y_q_scales) == rate_num:
        pass
    else:
        max_y_q_scale = p_frame_y_q_scales[0]
        min_y_q_scale = p_frame_y_q_scales[-1]
        p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_num)

        max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
        min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
        p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_num)

    # Выбор девайса
    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    if gpu_num > 0:  # TODO: add multiprocessing like in test_video.py
        device = "cuda"
    else:
        device = "cpu"

    # Инициализация FasterRCNN
    pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=pretrained_weights)
    rcnn = rcnn.to(device)
    rcnn.eval()
    original_bboxes = []

    # Загрузка весов модели для изображений и её дальнейшая инициализация
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict, strict=False)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    # Загрузка весов модели для видео и её дальнейшая инициализация
    p_state_dict = get_state_dict(args.model_path)
    video_net = DMC()
    video_net.load_state_dict(p_state_dict, strict=False)
    video_net = video_net.to(device)
    video_net.eval()

    # Подготовка аргументов для DCVC-HEM
    root_path = config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            obj_seq = []
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['i_frame_q_scale'] = i_frame_q_scales[rate_idx]
                cur_args['p_frame_y_q_scale'] = p_frame_y_q_scales[rate_idx]
                cur_args['p_frame_mv_y_q_scale'] = p_frame_mv_y_q_scales[rate_idx]
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra_period > 0:
                    cur_args['gop'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}_DMC_{rate_idx}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    device,
                    i_frame_net,
                    video_net,
                    cur_args)
                obj_seq.append(obj)
            objs.append(obj_seq)

            print(f'Getting original bound boxes for sequence {seq_name}')
            sub_dir_name = cur_args['video_path']
            cur_args['img_path'] = os.path.join(cur_args['dataset_path'], sub_dir_name)
            bboxes = get_original_bboxes(rcnn, cur_args, 'cuda')
            original_bboxes.append(bboxes)

    results = []
    for seq_index in range(len(objs)):
        print(f'Sequence: {seq_index + 1}')
        results_seq = []
        for index, obj in enumerate(objs[seq_index]):
            print(f'\tIteration: {index + 1}')
            result = obj.result()
            results_seq.append(result)
        results.append(results_seq)

    with torch.no_grad():
        metric = MeanAveragePrecision()
        for res_seq_index in range(len(results)):
            maps = []
            bpps = []
            for res_index, result in enumerate(results[res_seq_index]):
                decoded_bboxes = []
                target_bboxes = []
                for index in range(len(result['decoded'])):
                    if result['frame_type'][index]:
                        output = forward_rcnn(rcnn, result['decoded'][index])
                        decoded_bboxes.append(output)
                        target_bboxes.append(original_bboxes[res_seq_index][index])

                metric.update(decoded_bboxes, target_bboxes)
                map_metrics = metric.compute()
                maps.append(map_metrics['map_50'].item())
                bpps.append(result['ave_p_frame_bpp'])

            x = np.array(bpps)
            y = np.array(maps)
            plt.plot(x, y, 'o-', label='Original DCVC-HEM')
            plt.legend()
            plt.title('Object detection performance')
            plt.xlabel('bpp')
            plt.ylabel('mAP@0.5 (%)')
            plt.show()

    print('Test finished')


if __name__ == "__main__":
    main()

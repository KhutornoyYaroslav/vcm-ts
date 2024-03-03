import logging
import os

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from DCVC_HEM.src.utils.stream_helper import get_padding_size
from core.utils.tensorboard import add_best_and_worst_sample


def read_dataset_classes(cfg):
    metadata = os.path.join(cfg.DATASET.METADATA_PATH)
    classes = []
    with open(metadata) as f:
        for line in f.readlines():
            elements = line.split(': ')
            classes.append(int(elements[0]))
    return classes


def forward_yolo(yolo, x, labels_start_index=1):
    output = {}
    with torch.no_grad():
        pic_height = x.shape[2]
        pic_width = x.shape[3]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, p=32)
        x_padded = torch.nn.functional.pad(
            x,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        result = yolo(x_padded, imgsz=(x_padded.shape[2], x_padded.shape[3]), verbose=False)[0]  # batch = 1
        output['boxes'] = result.boxes.xyxy.cpu()
        output['labels'] = result.boxes.cls.cpu().type(torch.int64) + labels_start_index  # yolo names start from 0
        output['scores'] = result.boxes.conf.cpu()
        return output


def delete_unsupported_annotations(annotations, classes):
    for sub_annotations in annotations:
        for annotation in sub_annotations:
            mask = torch.ones(annotation['labels'].shape[0], dtype=torch.bool)
            for i in range(annotation['labels'].shape[0]):
                mask[i] = annotation['labels'][i].item() in classes
            annotation['boxes'] = annotation['boxes'][mask]
            annotation['labels'] = annotation['labels'][mask]
            annotation['scores'] = annotation['scores'][mask]


def eval_dataset(model, forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, cfg,
                 object_detection_loader=None, stage=0, perceptual_loss_key=None):
    logger = logging.getLogger("CORE.inference")

    # Iteration loop
    stats = {
        'loss_sum': 0,
        'perceptual_loss': 0,
        'bpp': 0,
        'mse_sum': 0,
        'psnr': 0,
        'mean_ap': [],
        'best_samples': [],
        'worst_samples': []
    }

    sample_count = 0
    n = len(cfg.SOLVER.LAMBDAS)
    best_samples = [[] for _ in range(n)]
    worst_samples = [[] for _ in range(n)]
    for data_entry in tqdm(data_loader):
        input, _ = data_entry  # (N, T, C, H, W)

        # Forward images
        with torch.no_grad():
            # Forward data to GPU
            input = input.cuda()

            # Do prediction
            outputs = model(forward_method,
                            input,
                            loss_dist_key,
                            loss_rate_keys,
                            p_frames=p_frames,
                            perceptual_loss_key=perceptual_loss_key,
                            is_train=False)

        stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (T-1) -> (1)
        stats['perceptual_loss'] += torch.mean(outputs['perceptual_loss'], -1).item()  # (T-1) -> (1)
        stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        stats['mse_sum'] += 0  # TODO:
        stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        sample_count += outputs['single_forwards']

        add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

    if object_detection_loader is not None and stage >= cfg.DATASET.OD_STAGE:
        yolo_detection = YOLO('pretrained/yolov8m.pt')
        yolo_detection = yolo_detection.cuda()
        classes = read_dataset_classes(cfg)
        output_annotations = [[] for _ in range(n)]
        source_annotations = [[] for _ in range(n)]
        for frame_idx, data_entry in enumerate(tqdm(object_detection_loader)):
            with torch.no_grad():
                input, annotation = data_entry  # (1, T, C, H, W)
                input = input.expand(n, -1, -1, -1, -1)  # (N, T, C, H, W)
                input = input.cuda()
                source_annotation = {
                    "boxes": annotation["boxes"][0],
                    "labels": annotation["labels"][0]
                }
                for i in range(n):
                    source_annotations[i].append(source_annotation)

                if frame_idx % cfg.DATASET.OD_GOP_SIZE == 0:
                    dpb = []
                    for i in range(n):
                        dpb.append({
                            "ref_frame": input[i],
                            "ref_feature": None,
                            "ref_y": None,
                            "ref_mv_y": None,
                        })

                    output = forward_yolo(yolo_detection, input[0])
                    for i in range(n):
                        output_annotations[i].append(output)
                else:
                    dpb = model('forward_simple',
                                input,
                                dpb=dpb)

                    torch.cuda.empty_cache()
                    for i in range(n):
                        input_yolo = dpb[i]["ref_frame"]  # (N, C, H, W)
                        input_yolo = input_yolo.clamp(0, 1)
                        output = forward_yolo(yolo_detection, input_yolo)
                        output_annotations[i].append(output)
                    torch.cuda.empty_cache()

        delete_unsupported_annotations(output_annotations, classes)
        metric_map = MeanAveragePrecision(compute_on_cpu=True, sync_on_compute=False, distributed_available_fn=None)
        for i in range(n):
            metric_map.update(output_annotations[i], source_annotations[i])
            map_metrics = metric_map.compute()
            mean_ap = map_metrics['map_50'].item() * 100
            stats['mean_ap'].append(mean_ap)

    # Return results
    stats['loss_sum'] /= sample_count
    stats['perceptual_loss'] /= sample_count
    stats['bpp'] /= sample_count
    stats['mse_sum'] /= sample_count
    stats['psnr'] /= sample_count
    stats['best_samples'] = best_samples
    stats['worst_samples'] = worst_samples
    if not stats['mean_ap']:
        stats['mean_ap'] = [0] * n
    stats['mean_ap'] = np.asarray(stats['mean_ap'])

    return stats

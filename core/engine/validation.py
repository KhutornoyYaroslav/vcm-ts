import logging
import os

import numpy as np
import torch
import torchvision
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.utils.common import interpolate_log
from DCVC_HEM.src.utils.stream_helper import get_padding_size, get_state_dict
from core.engine.losses import FasterRCNNFPNPerceptualLoss, FasterRCNNResNetPerceptualLoss
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


def forward_rcnn(rcnn, x):
    with torch.no_grad():
        output = rcnn(x)[0]  # batch = 1
        output['boxes'] = output['boxes'].cpu()
        output['labels'] = output['labels'].cpu()
        output['scores'] = output['scores'].cpu()
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
                 object_detection_loader=None, stage=0, perceptual_loss=False, i_frame_net=None, i_frame_q_scales=None):
    logger = logging.getLogger("CORE.inference")

    # Iteration loop
    n = len(cfg.SOLVER.LAMBDAS)
    stats = {
        'loss_sum': 0,
        'dist': 0,
        'p_dist': 0,
        'bpp': np.zeros(n),
        'psnr': np.zeros(n),
        'mean_ap': [],
        'best_samples': [],
        'worst_samples': []
    }

    sample_count = 0
    best_samples = [[] for _ in range(n)]
    worst_samples = [[] for _ in range(n)]
    for data_entry in tqdm(data_loader):
        input, target = data_entry  # (N, T, C, H, W)

        # Forward images
        with torch.no_grad():
            # Forward data to GPU
            input = input.cuda()
            target = target.cuda()

            # Do prediction
            outputs = model(forward_method,
                            input,
                            target,
                            loss_dist_key,
                            loss_rate_keys,
                            p_frames=p_frames,
                            perceptual_loss=perceptual_loss,
                            is_train=False,
                            i_frame_net=i_frame_net,
                            i_frame_q_scales=i_frame_q_scales)

        stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (T-1) -> (1)
        stats['dist'] += torch.mean(torch.sum(outputs['dist'], -1)).item()  # (T-1) -> (1)
        stats['p_dist'] += torch.mean(torch.sum(outputs['p_dist'], -1)).item()  # (T-1) -> (1)
        stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        sample_count += outputs['single_forwards']

        add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

    if i_frame_net is None:
        rate_count = len(cfg.SOLVER.LAMBDAS)
        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt('pretrained/acmmm2022_image_psnr.pth')
        if len(i_frame_q_scales) == rate_count:
            pass
        else:
            max_q_scale = i_frame_q_scales[0]
            min_q_scale = i_frame_q_scales[-1]
            i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)

        i_state_dict = get_state_dict('pretrained/acmmm2022_image_psnr.pth')
        i_frame_net = IntraNoAR()
        i_frame_net.load_state_dict(i_state_dict, strict=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()
    if object_detection_loader is not None and stage >= cfg.DATASET.OD_STAGE:
        if (isinstance(model.perceptual_loss, FasterRCNNFPNPerceptualLoss) or
                isinstance(model.perceptual_loss, FasterRCNNResNetPerceptualLoss)):
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(min_size=1088, max_size=1920)
            detector.load_state_dict(torch.load('pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
            detector.cuda()
            detector.eval()
        else:
            detector = YOLO('pretrained/yolov8m.pt')
            detector = detector.cuda()
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
                        if i_frame_net is not None:
                            out_i = i_frame_net(input[i], i_frame_q_scales[i])
                            dpb.append({
                                "ref_frame": out_i["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None
                            })
                        else:
                            dpb.append({
                                "ref_frame": input[i],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            })
                    torch.cuda.empty_cache()
                else:
                    dpb = model('forward_simple',
                                input,
                                dpb=dpb)
                    torch.cuda.empty_cache()

                for i in range(n):
                    input_to_detect = dpb[i]["ref_frame"]  # (N, C, H, W)
                    input_to_detect = input_to_detect.clamp(0, 1)
                    if (isinstance(model.perceptual_loss, FasterRCNNFPNPerceptualLoss) or
                            isinstance(model.perceptual_loss, FasterRCNNResNetPerceptualLoss)):
                        output = forward_rcnn(detector, input_to_detect)
                    else:
                        output = forward_yolo(detector, input_to_detect)
                    output_annotations[i].append(output)
                    torch.cuda.empty_cache()

        delete_unsupported_annotations(output_annotations, classes)
        metric_map = MeanAveragePrecision(compute_on_cpu=True, sync_on_compute=False, distributed_available_fn=None)
        for i in range(n):
            metric_map.update(output_annotations[i], source_annotations[i])
            map_metrics = metric_map.compute()
            mean_ap = map_metrics['map'].item() * 100
            stats['mean_ap'].append(mean_ap)

    # Return results
    if sample_count == 0:
        sample_count = 1
    stats['loss_sum'] /= sample_count
    stats['dist'] /= sample_count
    stats['p_dist'] /= sample_count
    stats['bpp'] /= sample_count
    stats['psnr'] /= sample_count
    stats['best_samples'] = best_samples
    stats['worst_samples'] = worst_samples
    if not stats['mean_ap']:
        stats['mean_ap'] = [0] * n
    stats['mean_ap'] = np.asarray(stats['mean_ap'])

    return stats

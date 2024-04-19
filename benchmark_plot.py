import argparse
import itertools
import json
import os
import re
from difflib import SequenceMatcher
from glob import glob

import cv2
import jaro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from jiwer import cer
from paddleocr import PaddleOCR
from pytorch_msssim import MS_SSIM
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from DCVC_HEM.src.utils.png_reader import PNGReader
from DCVC_HEM.src.utils.stream_helper import get_padding_size


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def get_psnr(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def delete_unsupported_annotations(annotations, classes):
    for key in annotations.keys():
        if key == "yolo_lp_detection" or key == "ocr_result" or len(annotations[key]) == 0:
            continue

        for annotation in annotations[key]:
            mask = torch.ones(annotation['labels'].shape[0], dtype=torch.bool)
            for i in range(annotation['labels'].shape[0]):
                mask[i] = annotation['labels'][i].item() in classes
            annotation['boxes'] = annotation['boxes'][mask]
            annotation['labels'] = annotation['labels'][mask]
            annotation['scores'] = annotation['scores'][mask]


def read_object_detection(annotation, device):
    boxes = []
    labels = []
    with open(annotation) as f:
        for line in f.readlines():
            elements = list(map(int, line.split()))
            boxes.append(elements[1:5])
            labels.append(elements[0])
    target = dict(
        boxes=torch.tensor(boxes, dtype=torch.float32, device=device),
        labels=torch.tensor(labels, dtype=torch.int64, device=device)
    )
    return target


def read_license_detection(annotation, device):
    boxes = []
    labels = []
    with open(annotation) as f:
        for line in f.readlines():
            elements = list(map(int, line.split()))
            boxes.append(elements)
            labels.append(0)
    target = dict(
        boxes=torch.tensor(boxes, dtype=torch.float32, device=device),
        labels=torch.tensor(labels, dtype=torch.int64, device=device)
    )
    return target


def read_license_recognition(annotation, device):
    boxes = []
    texts = []
    with open(annotation) as f:
        for line in f.readlines():
            elements = line.split()
            boxes.append(list(map(int, elements[1:5])))
            texts.append(elements[0])
    target = dict(
        boxes=torch.tensor(boxes, dtype=torch.float32, device=device),
        texts=texts
    )
    return target


def read_dataset(config,
                 rcnn,
                 yolo_detection,
                 yolo_lp_detection,
                 device: str):
    dataset = {}
    sequences = config["sequences"]
    for sequence in sequences:
        print(f'Sequence: {sequence["name"]}')
        sequence_path = os.path.join(config["dataset_dir"], sequence["name"])
        images = []
        annotations = {}
        images_folder = os.path.join(sequence_path, "images")
        src_reader = PNGReader(images_folder)
        source_images = sorted(glob(os.path.join(images_folder, "*.png")))
        for annotation_type in sequence["annotation_types"]:
            annotations[annotation_type] = []
            annotations_folder = os.path.join(sequence_path, annotation_type)
            source_annotations = sorted(glob(os.path.join(annotations_folder, "*.txt")))
            assert len(source_images) == len(source_annotations)
            for annotation in tqdm(source_annotations):
                if annotation_type == "object_detection":
                    target = read_object_detection(annotation, device)
                elif annotation_type == "license_detection":
                    target = read_license_detection(annotation, device)
                elif annotation_type == "license_recognition":
                    target = read_license_recognition(annotation, device)
                else:
                    raise AttributeError("Invalid annotation type in config file")
                annotations[annotation_type].append(target)

        for _ in tqdm(source_images):
            rgb = src_reader.read_one_frame(src_format="rgb")
            image = np_image_to_tensor(rgb)
            image = image.to(device)
            images.append(image)

        metadata = os.path.join(sequence_path, "metadata.txt")
        classes = []
        class_names = []
        with open(metadata) as f:
            for line in f.readlines():
                elements = line.split(': ')
                classes.append(int(elements[0]))
                class_names.append(elements[1].strip())
        dataset[sequence["name"]] = dict(images=images,
                                         annotations=annotations,
                                         classes=classes,
                                         class_names=class_names)

        annotation_types = annotations.keys()
        mean_ap = 0
        if "object_detection" in annotation_types or "license_detection" in annotation_types:
            origin_annotations = {
                'rcnn': [],
                'yolo_detection': [],
                'yolo_lp_detection': []
            }
            _, _, height, width = image.shape
            rcnn.transform.min_size = (height,)
            rcnn.transform.max_size = width
            for image in tqdm(images):
                if "object_detection" in annotation_types:
                    rcnn_output = forward_rcnn(rcnn, image.cuda())
                    yolo_detection_output = forward_yolo(yolo_detection, image.cuda())
                    origin_annotations['rcnn'].append(rcnn_output)
                    origin_annotations['yolo_detection'].append(yolo_detection_output)
                elif "license_detection" in annotation_types:
                    yolo_lp_detection_output = forward_yolo(yolo_lp_detection, image.cuda(), 0)
                    origin_annotations['yolo_lp_detection'].append(yolo_lp_detection_output)

            delete_unsupported_annotations(origin_annotations, dataset[sequence["name"]]['classes'])
            mean_ap = calculate_mean_ap(origin_annotations, dataset, sequence["name"])

        dataset[sequence["name"]]["mean_ap"] = mean_ap

    return dataset


def forward_rcnn(rcnn, x):
    with torch.no_grad():
        output = rcnn(x)[0]  # batch = 1
        output['boxes'] = output['boxes'].cpu()
        output['labels'] = output['labels'].cpu()
        output['scores'] = output['scores'].cpu()
        return output


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


def forward_ocr(ocr, image, boxes):
    result = []
    for i in range(boxes.shape[0]):
        x, y, w, h = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        plate = np.transpose(image[0, :, y:h, x:w].cpu().numpy(), (1, 2, 0)) * 255
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        result_ocr = ocr.ocr(plate, cls=False, det=False)
        result_ocr = re.sub('[^A-Z0-9]+', '', result_ocr[0][0][0].upper())
        result.append(result_ocr)

    return result


def calculate_accuracy_symbolically(text1, text2):
    num_matches = 0
    if len(text1) == 0 and len(text2) == 1:
        return 1
    if len(text1) == 0:
        return 0
    for symbol1, symbol2 in zip(text1, text2):
        if symbol1 == symbol2:
            num_matches += 1
    accuracy = num_matches / len(text1)
    return accuracy


def calculate_ocr_metrics(dataset_annotations, annotations):
    full = 0
    symbol_equal_len = 0
    symbol_nonequal_len = 0
    RO = 0
    JW = 0
    cer_value = 0
    assert len(dataset_annotations) == len(annotations), "Annotations size for calculate ocr metrics should be the same"
    for dataset_annotation, annotation in zip(dataset_annotations, annotations):
        # Full text match
        if dataset_annotation == annotation:
            full += 1

        # Symbolically test weighted match, equal length
        if len(dataset_annotation) == len(annotation):
            symbol_equal_len += calculate_accuracy_symbolically(dataset_annotation, annotation)

        # Symbolically test weighted match, non-equal length
        symbol_nonequal_len += calculate_accuracy_symbolically(dataset_annotation, annotation)

        # Ratcliff and Obershelp algorithm match
        s = SequenceMatcher(None, dataset_annotation, annotation)
        RO += s.ratio()

        # Jaro-Winkler algorithm match
        JW += jaro.jaro_winkler_metric(dataset_annotation, annotation)

        # CER (1 - CER value) match
        cer_value += 1 - cer(dataset_annotation, annotation)

    length = len(dataset_annotations)
    return {
        "full": full / length * 100,
        "symbol_equal_len": symbol_equal_len / length * 100,
        "symbol_nonequal_len": symbol_nonequal_len / length * 100,
        "RO": RO / length * 100,
        "JW": JW / length * 100,
        "cer": cer_value / length * 100
    }


def calculate_mean_ap(annotations, dataset, video_name):
    metric_map = MeanAveragePrecision(class_metrics=True)
    mean_ap = {}
    for model in annotations.keys():
        if len(annotations[model]) == 0:
            continue

        if model == 'ocr_result':
            continue
        elif model in ['rcnn', 'yolo_detection']:
            dataset_annotations = dataset[video_name]['annotations']['object_detection']
        elif model == 'yolo_lp_detection':
            dataset_annotations = dataset[video_name]['annotations']['license_detection']
        else:
            raise RuntimeError("Invalid model type for calculate metrics")
        mean_ap[model] = {}
        metric_map.update(annotations[model], dataset_annotations)
        map_metrics = metric_map.compute()
        for class_mean_ap, class_id in zip(map_metrics['map_per_class'], map_metrics['classes']):
            class_name = dataset[video_name]['class_names'][dataset[video_name]['classes'].index(class_id)]
            mean_ap[model][class_name] = class_mean_ap.item() * 100
        model_mean_ap = map_metrics['map_50'].item()
        mean_ap[model]['map_50'] = model_mean_ap * 100

    return mean_ap


def calculate_metrics(dataset,
                      images,
                      annotations,
                      video_name: str,
                      use_ms_ssim: bool):
    metric_ssim = MS_SSIM(data_range=1.0, size_average=False)
    dataset_images = dataset[video_name]['images']
    ocr_results = {}
    if 'ocr_result' in annotations.keys() and len(annotations['ocr_result']) != 0:
        dataset_annotations = dataset[video_name]['annotations']['license_recognition']
        merged_annotations = list(itertools.chain.from_iterable(annotations['ocr_result']))
        merged_dataset_annotations = map(lambda d: d['texts'], dataset_annotations)
        merged_dataset_annotations = list(itertools.chain.from_iterable(merged_dataset_annotations))
        ocr_results = calculate_ocr_metrics(merged_dataset_annotations, merged_annotations)
    mean_ap = calculate_mean_ap(annotations, dataset, video_name)

    psnr_list = []
    ssim_list = []
    for dataset_image, image in tqdm(zip(dataset_images, images), total=len(dataset_images)):
        psnr = get_psnr(image, dataset_image)
        if use_ms_ssim:
            ssim = metric_ssim(dataset_image, image)
            ssim_list.append(ssim)
        psnr_list.append(psnr)

    psnr = np.mean(np.array(psnr_list))
    if use_ms_ssim:
        ssim = torch.stack(ssim_list, -1)
        ssim = torch.mean(ssim, -1).item()
    else:
        ssim = None
    return mean_ap, ocr_results, psnr, ssim


def get_metrics(decod_dir: str,
                rcnn,
                yolo_detection,
                yolo_lp_detection,
                ocr,
                dataset,
                device: str,
                use_ms_ssim: bool):
    metrics = {}
    model_folders = [f for f in os.scandir(decod_dir) if f.is_dir()]
    dataset_videos = dataset.keys()

    for model_folder in model_folders:
        print(f'Calculate metrics for {model_folder.name}')
        metrics[model_folder.name] = {}
        video_folders = [f for f in os.scandir(model_folder) if f.is_dir()]

        for video_folder in video_folders:
            if video_folder.name not in dataset_videos:
                continue
            print(f'\tCalculate metrics for video {video_folder.name}')
            metrics[model_folder.name][video_folder.name] = []
            images_folders = [f for f in os.scandir(video_folder) if f.is_dir()]
            images_folders.sort(key=lambda folder: folder.name)
            annotation_types = dataset[video_folder.name]["annotations"].keys()

            for images_folder in images_folders:
                metrics_json = images_folder.path + '_metrics.json'
                if os.path.exists(metrics_json):
                    with open(metrics_json) as f:
                        metrics_info = json.load(f)
                    metrics[model_folder.name][video_folder.name].append(metrics_info)
                    print(f'\t\tRead metrics for {images_folder.name} from json')
                    continue

                info_json = images_folder.path + '.json'
                with open(info_json) as f:
                    seq_info = json.load(f)
                bpp = seq_info['avg_bpp']
                gop = seq_info['gop']
                frame_bpp = []
                if 'frame_bpp' in seq_info.keys():
                    frame_bpp = seq_info['frame_bpp']
                print(f'\t\tCalculate metrics for sequence with bpp = {bpp} and gop = {gop}')
                images = []
                annotations = {
                    'rcnn': [],
                    'yolo_detection': [],
                    'yolo_lp_detection': [],
                    'ocr_result': []
                }
                source_images = sorted(glob(os.path.join(images_folder, "*.png")))

                print(f'\t\tObjects detection and recognition')
                src_reader = PNGReader(images_folder.path)
                for i in tqdm(range(len(source_images))):
                    rgb = src_reader.read_one_frame(src_format="rgb")
                    image = np_image_to_tensor(rgb)
                    image = image.to(device)

                    if "object_detection" in annotation_types:
                        _, _, height, width = image.shape
                        rcnn.transform.min_size = (height,)
                        rcnn.transform.max_size = width
                        rcnn_output = forward_rcnn(rcnn, image)
                        yolo_detection_output = forward_yolo(yolo_detection, image)
                        annotations['rcnn'].append(rcnn_output)
                        annotations['yolo_detection'].append(yolo_detection_output)
                    elif "license_detection" in annotation_types:
                        yolo_lp_detection_output = forward_yolo(yolo_lp_detection, image, 0)
                        annotations['yolo_lp_detection'].append(yolo_lp_detection_output)
                    elif "license_recognition" in annotation_types:
                        boxes = dataset[video_folder.name]["annotations"]["license_recognition"][i]["boxes"]
                        ocr_result = forward_ocr(ocr, image, boxes)
                        annotations['ocr_result'].append(ocr_result)

                    image = image.cpu()
                    torch.cuda.empty_cache()
                    images.append(image)

                delete_unsupported_annotations(annotations, dataset[video_folder.name]['classes'])

                print(f'\t\tCalculate metrics')
                mean_ap, ocr_results, psnr, ssim = calculate_metrics(dataset, images, annotations, video_folder.name, use_ms_ssim)
                metrics_info = dict(
                    mean_ap=mean_ap,
                    ocr_results=ocr_results,
                    psnr=psnr,
                    ssim=ssim,
                    bpp=bpp,
                    frame_bpp=frame_bpp,
                    gop=gop,
                    quality=images_folder.name
                )
                metrics[model_folder.name][video_folder.name].append(metrics_info)
                with open(metrics_json, 'w') as fp:
                    json.dump(metrics_info, fp)

    return metrics


def plot_graphs(metrics, dataset, out_path: str, use_ms_ssim: bool):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    os.makedirs(out_path, exist_ok=True)

    for video in videos:
        detection_models = sorted(list(metrics[codecs[0]][video][0]['mean_ap'].keys()))
        for detection_model in detection_models:
            plt.figure(figsize=(16, 9))
            orig_map = dataset[video]['mean_ap'][detection_model]['map_50']
            map_loss_1 = orig_map - 1
            map_loss_2 = orig_map - 2
            plt.axhline(y=orig_map, color='k', linestyle='dashed',
                        label=f'Original performance ({orig_map:.2f}%)')
            plt.axhline(y=map_loss_1, color='gray', linestyle='dashdot', label='1% mAP loss')
            plt.axhline(y=map_loss_2, color='gray', linestyle='dashdot', label='2% mAP loss')
            for codec in codecs:
                bpps = [info['bpp'] for info in metrics[codec][video]]
                maps = [info['mean_ap'][detection_model]['map_50'] for info in metrics[codec][video]]
                x = np.array(bpps)
                y = np.array(maps)
                plt.plot(x, y, 'o-', label=codec)
            plt.legend()
            plt.grid()
            plt.title(f'Object detection performance on {detection_model} for {video}')
            plt.xlabel('bpp')
            plt.ylabel('mAP@0.5 (%)')
            plt.savefig(os.path.join(out_path, f"detection_model_{detection_model}_{video}.png"))

            class_names = sorted(list(metrics[codecs[0]][video][0]['mean_ap'][detection_model].keys()))
            class_names.remove('map_50')
            for class_name in class_names:
                plt.figure(figsize=(16, 9))
                orig_map = dataset[video]['mean_ap'][detection_model][class_name]
                map_loss_1 = orig_map - 1
                map_loss_2 = orig_map - 2
                plt.axhline(y=orig_map, color='k', linestyle='dashed',
                            label=f'Original performance ({orig_map:.2f}%)')
                plt.axhline(y=map_loss_1, color='gray', linestyle='dashdot', label='1% mAP loss')
                plt.axhline(y=map_loss_2, color='gray', linestyle='dashdot', label='2% mAP loss')
                for codec in codecs:
                    bpps = [info['bpp'] for info in metrics[codec][video]]
                    maps = [info['mean_ap'][detection_model][class_name] for info in metrics[codec][video]]
                    x = np.array(bpps)
                    y = np.array(maps)
                    plt.plot(x, y, 'o-', label=codec)
                plt.legend()
                plt.grid()
                plt.title(f'Object detection performance for class {class_name} on {detection_model} for {video}')
                plt.xlabel('bpp')
                plt.ylabel('mAP (%)')
                plt.savefig(os.path.join(out_path, f"detection_model_{class_name}_{detection_model}_{video}.png"))

        matchers = sorted(list(metrics[codecs[0]][video][0]['ocr_results'].keys()))
        for matcher in matchers:
            plt.figure(figsize=(16, 9))
            for codec in codecs:
                bpps = [info['bpp'] for info in metrics[codec][video]]
                match_values = [info['ocr_results'][matcher] for info in metrics[codec][video]]
                x = np.array(bpps)
                y = np.array(match_values)
                plt.plot(x, y, 'o-', label=codec)
            plt.legend()
            plt.grid()
            plt.title(f'Text matching on {matcher} for {video}')
            plt.xlabel('bpp')
            plt.ylabel('Metric value, %')
            plt.savefig(os.path.join(out_path, f"text_match_{matcher}_{video}.png"))

        plt.figure(figsize=(16, 9))
        for codec in codecs:
            bpps = [info['bpp'] for info in metrics[codec][video]]
            psnrs = [info['psnr'] for info in metrics[codec][video]]
            x = np.array(bpps)
            y = np.array(psnrs)
            plt.plot(x, y, 'o-', label=codec)
        plt.legend()
        plt.grid()
        plt.title(f'Rate and distortion curves (PSNR) for {video}')
        plt.xlabel('bpp')
        plt.ylabel('PSNR (db)')
        plt.savefig(os.path.join(out_path, f"psnr_{video}.png"))

        if use_ms_ssim:
            plt.figure(figsize=(16, 9))
            for codec in codecs:
                bpps = [info['bpp'] for info in metrics[codec][video]]
                ssims = [info['ssim'] for info in metrics[codec][video]]
                x = np.array(bpps)
                y = np.array(ssims)
                plt.plot(x, y, 'o-', label=codec)
            plt.legend()
            plt.grid()
            plt.title(f'Rate and distortion curves (MS_SSIM) for {video}')
            plt.xlabel('bpp')
            plt.ylabel('MS-SSIM')
            plt.savefig(os.path.join(out_path, f"ms-ssim_{video}.png"))

        for codec in codecs:
            plot_flag = True
            plt.figure(figsize=(16, 9))
            for quality in metrics[codec][video]:
                if quality['frame_bpp']:
                    bpps = quality['frame_bpp']
                    frames_count = range(len(bpps))
                    x = np.array(frames_count)
                    y = np.array(bpps)
                    plt.plot(x, y, 'o-', label=quality['quality'])
                else:
                    plot_flag = False
                    break

            if plot_flag:
                plt.yscale('log')
                plt.legend()
                plt.grid()
                plt.title(f'Bpp per frame for codec {codec} and video {video}')
                plt.xlabel('frame')
                plt.ylabel('bpp')
                plt.savefig(os.path.join(out_path, f"bpp_{codec}_{video}.png"))


def main():
    parser = argparse.ArgumentParser(description='Benchmark graph plotting')
    parser.add_argument('--config', dest='config', type=str,
                        default="benchmark_config_plot.json",
                        help="Config for benchmark plot")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print('Reading dataset')
    pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=pretrained_weights,
                                                                   min_size=1088, max_size=1920)
    rcnn = rcnn.to(config["device"])
    rcnn.eval()

    yolo_detection = YOLO('pretrained/yolov8m.pt')
    yolo_detection = yolo_detection.to(config["device"])

    yolo_lp_detection = YOLO('pretrained/yolov8-lp.pt')
    yolo_lp_detection = yolo_lp_detection.to(config["device"])

    # cpu for optimization video memory
    dataset = read_dataset(config, rcnn, yolo_detection, yolo_lp_detection, 'cpu')

    ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_algorithm='SVTR_LCNet')

    metrics = get_metrics(config["decoded_dir"],
                          rcnn,
                          yolo_detection,
                          yolo_lp_detection,
                          ocr,
                          dataset,
                          config["device"],
                          config["ms_ssim"])

    plot_graphs(metrics, dataset, config["out_path"], config["ms_ssim"])


if __name__ == "__main__":
    main()

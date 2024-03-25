import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torchvision
from pytorch_msssim import MS_SSIM
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    with open(annotation) as f:
        for line in f.readlines():
            elements = list(map(int, line.split()))
            boxes.append(elements)
    target = dict(
        boxes=torch.tensor(boxes, dtype=torch.float32, device=device)
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
                 device: str):
    dataset = {}
    metadata = os.path.join(config["dataset_dir"], "metadata.txt")
    classes = []
    with open(metadata) as f:
        for line in f.readlines():
            elements = line.split(': ')
            classes.append(int(elements[0]))
    dataset['classes'] = classes

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

        dataset[sequence["name"]] = dict(images=images, annotations=annotations)

    return dataset


def forward_rcnn(rcnn, x):
    with torch.no_grad():
        output = rcnn(x)[0]  # batch = 1
        output['boxes'] = output['boxes'].cpu()
        output['labels'] = output['labels'].cpu()
        output['scores'] = output['scores'].cpu()
        return output


def forward_yolo(yolo, x):
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
        output['labels'] = result.boxes.cls.cpu().type(torch.int64) + 1  # yolo names start from 0
        output['scores'] = result.boxes.conf.cpu()
        return output


def calculate_metrics(dataset,
                      images,
                      annotations,
                      video_name: str,
                      use_ms_ssim: bool):
    metric_map = MeanAveragePrecision()
    metric_ssim = MS_SSIM(data_range=1.0, size_average=False)

    dataset_images = dataset[video_name]['images']
    dataset_annotations = dataset[video_name]['annotations']['object_detection']
    # TODO: add other annotation types
    mean_ap = {}
    for model in annotations.keys():
        metric_map.update(annotations[model], dataset_annotations)
        map_metrics = metric_map.compute()
        model_mean_ap = map_metrics['map_50'].item()
        mean_ap[model] = model_mean_ap

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
    return mean_ap, psnr, ssim


def get_metrics(decod_dir: str,
                rcnn,
                yolo,
                dataset,
                device: str,
                use_ms_ssim: bool):
    metrics = {}
    model_folders = [f for f in os.scandir(decod_dir) if f.is_dir()]

    for model_folder in model_folders:
        print(f'Calculate metrics for {model_folder.name}')
        metrics[model_folder.name] = {}
        video_folders = [f for f in os.scandir(model_folder) if f.is_dir()]

        for video_folder in video_folders:
            if "object_detection" not in dataset[video_folder.name]["annotations"].keys():
                continue
            print(f'\tCalculate metrics for video {video_folder.name}')
            metrics[model_folder.name][video_folder.name] = []
            images_folders = [f for f in os.scandir(video_folder) if f.is_dir()]
            images_folders.sort(key=lambda folder: folder.name)

            for images_folder in images_folders:
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
                    'yolo': []
                }
                source_images = sorted(glob(os.path.join(images_folder, "*.png")))

                print(f'\t\tObjects detection')
                src_reader = PNGReader(images_folder.path)
                for _ in tqdm(range(len(source_images))):
                    rgb = src_reader.read_one_frame(src_format="rgb")
                    image = np_image_to_tensor(rgb)
                    image = image.to(device)

                    rcnn_output = forward_rcnn(rcnn, image)
                    yolo_output = forward_yolo(yolo, image)
                    image = image.cpu()
                    torch.cuda.empty_cache()
                    images.append(image)
                    annotations['rcnn'].append(rcnn_output)
                    annotations['yolo'].append(yolo_output)

                delete_unsupported_annotations(annotations, dataset['classes'])

                print(f'\t\tCalculate metrics')
                mean_ap, psnr, ssim = calculate_metrics(dataset, images, annotations, video_folder.name, use_ms_ssim)
                metrics_info = dict(
                    mean_ap=mean_ap,
                    psnr=psnr,
                    ssim=ssim,
                    bpp=bpp,
                    frame_bpp=frame_bpp,
                    gop=gop,
                    quality=images_folder.name
                )
                metrics[model_folder.name][video_folder.name].append(metrics_info)

    return metrics


def plot_graphs(metrics, out_path: str, use_ms_ssim: bool):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    detection_models = sorted(list(metrics[codecs[0]][videos[0]][0]['mean_ap'].keys()))
    os.makedirs(out_path, exist_ok=True)

    for video in videos:
        for detection_model in detection_models:
            for codec in codecs:
                bpps = [info['bpp'] for info in metrics[codec][video]]
                maps = [info['mean_ap'][detection_model] for info in metrics[codec][video]]
                x = np.array(bpps)
                y = np.array(maps)
                plt.plot(x, y, 'o-', label=codec)
            plt.legend()
            plt.grid()
            plt.title(f'Object detection performance on {detection_model} for {video}')
            plt.xlabel('bpp')
            plt.ylabel('mAP@0.5 (%)')
            plt.savefig(os.path.join(out_path, f"detection_model_{detection_model}_{video}.png"))
            plt.show()

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
        plt.show()

        if use_ms_ssim:
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
            plt.show()

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
                plt.show()


def main():
    parser = argparse.ArgumentParser(description='Benchmark graph plotting')
    parser.add_argument('--config', dest='config', type=str,
                        default="benchmark_config_plot.json",
                        help="Config for benchmark plot")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print('Reading dataset')
    dataset = read_dataset(config, 'cpu')  # cpu for optimization video memory

    pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=pretrained_weights)
    rcnn = rcnn.to(config["device"])
    rcnn.eval()

    yolo = YOLO('yolov8n.pt')
    yolo = yolo.to(config["device"])

    metrics = get_metrics(config["decoded_dir"], rcnn, yolo, dataset, config["device"], config["ms_ssim"])

    plot_graphs(metrics, config["out_path"], config["ms_ssim"])


if __name__ == "__main__":
    main()

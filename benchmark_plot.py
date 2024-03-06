import argparse
import os
from glob import glob

import numpy as np
import torch
import torchvision
from pytorch_msssim import MS_SSIM
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from DCVC_HEM.src.utils.png_reader import PNGReader


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def read_dataset(dataset_dir: str,
                 device: str):
    dataset = {}
    seq_folders = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    for seq_folder in seq_folders:
        print(f'Sequence: {seq_folder.name}')
        images = []
        annotations = []
        images_folder = os.path.join(seq_folder.path, "images")
        annotations_folder = os.path.join(seq_folder.path, "annotations")
        src_reader = PNGReader(images_folder)
        source_images = sorted(glob(os.path.join(images_folder, "*.png")))
        source_annotations = sorted(glob(os.path.join(annotations_folder, "*.txt")))
        assert len(source_images) == len(source_annotations)
        for annotation in tqdm(source_annotations):
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
            annotations.append(target)

            rgb = src_reader.read_one_frame(src_format="rgb")
            image = np_image_to_tensor(rgb)
            image = image.to(device)
            images.append(image)

        dataset[seq_folder.name] = dict(images=images, annotations=annotations)

    return dataset


def forward_rcnn(rcnn, x):
    with torch.no_grad():
        output = rcnn(x)[0]  # batch = 1
        output['boxes'] = output['boxes'].cpu()
        output['labels'] = output['labels'].cpu()
        output['scores'] = output['scores'].cpu()
        return output


def calculate_metrics(dataset,
                      images,
                      annotations,
                      video_name: str,
                      bpp: float,
                      gop: int):
    metric_map = MeanAveragePrecision()
    metric_mse = nn.MSELoss(reduction='none')
    metric_ssim = MS_SSIM(data_range=1.0, size_average=False)

    dataset_images = [image for i, image in enumerate(dataset[video_name]['images']) if i % gop != 0]
    dataset_annotations = [anno for i, anno in enumerate(dataset[video_name]['annotations']) if i % gop != 0]
    metric_map.update(annotations, dataset_annotations)
    map_metrics = metric_map.compute()
    mean_ap = map_metrics['map_50'].item()

    mse_list = []
    ssim_list = []
    for dataset_image, image in zip(dataset_images, images):
        _, _, H, W = image.size()
        pixel_num = H * W
        mse = metric_mse(dataset_image, image)
        ssim = metric_ssim(dataset_image, image)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        mse = torch.squeeze(mse)
        mse_list.append(mse)
        ssim_list.append(ssim)

    mse = torch.stack(mse_list, -1)
    mse = torch.mean(mse, -1)
    psnr = 10 * np.log10(1.0 / mse)
    ssim = torch.stack(ssim_list, -1)
    ssim = torch.mean(ssim, -1)
    return mean_ap, psnr, ssim


def get_metrics(decod_dir: str,
                rcnn,
                dataset,
                device: str):
    metrics = {}
    metric = MeanAveragePrecision()
    model_folders = [f for f in os.scandir(decod_dir) if f.is_dir()]

    for model_folder in model_folders:
        print(f'Calculate metrics for {model_folder.name}')
        metrics[model_folder.name] = {}
        video_folders = [f for f in os.scandir(model_folder) if f.is_dir()]

        for video_folder in video_folders:
            print(f'\tCalculate metrics for video {video_folder.name}')
            metrics[model_folder.name][video_folder.name] = []
            images_folders = [f for f in os.scandir(video_folder) if f.is_dir()]

            for images_folder in images_folders:
                seq_info = images_folder.name.split('_')
                bpp = float(seq_info[0])
                gop = int(seq_info[1])
                print(f'\tCalculate metrics for sequence with bpp = {bpp} and gop = {gop}')
                images = []
                annotations = []
                source_images = sorted(glob(os.path.join(images_folder, "*.png")))

                for index in tqdm(range(len(source_images))):
                    if index % gop != 0:
                        src_reader = PNGReader(images_folder.path)
                        rgb = src_reader.read_one_frame(src_format="rgb")
                        image = np_image_to_tensor(rgb)
                        image = image.to(device)

                        output = forward_rcnn(rcnn, image)
                        image = image.cpu()
                        torch.cuda.empty_cache()
                        images.append(image)
                        annotations.append(output)

                calculate_metrics(dataset, images, annotations, video_folder.name, bpp, gop)
                metrics[model_folder.name][video_folder.name].append(predictions_seq)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Benchmark graph plotting')
    parser.add_argument('--dataset-dir', dest='dataset_dir', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/dataset",
                        help="Path to dataset directory")
    parser.add_argument('--decod-dir', dest='decod_dir', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts-copy/outputs",
                        help="Path to results directory of decoding stage")
    parser.add_argument('--device', dest='device', type=str,
                        default="cuda",
                        help="Device for tensors")
    parser.add_argument('--out-path', dest='out_path', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark",
                        help="Path to output dir with graphs")
    args = parser.parse_args()

    print('Reading dataset')
    dataset = read_dataset(args.dataset_dir, 'cpu')  # cpu for optimization video memory

    pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=pretrained_weights)
    rcnn = rcnn.to(args.device)
    rcnn.eval()

    get_metrics(args.decod_dir, rcnn, dataset, args.device)


if __name__ == "__main__":
    main()

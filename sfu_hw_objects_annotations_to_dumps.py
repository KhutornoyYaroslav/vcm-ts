import argparse
import os
import shutil
from glob import glob

from tqdm import tqdm

COCO_CLASS_DICT = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
}


def convert_annotations(dataset_dir: str,
                        output_dir: str,
                        filename_template="im%05d.txt"):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    video_classes = [f for f in os.scandir(dataset_dir) if f.is_dir()]
    video_classes.sort(key=lambda x: x.name)
    for video_class in video_classes:
        video_folders = [f for f in os.scandir(video_class.path) if f.is_dir()]
        video_folders.sort(key=lambda x: x.name)
        for video_folder in video_folders:
            source_annotations = sorted(glob(os.path.join(video_folder.path, "*.txt")))
            out_dir = os.path.join(output_dir, video_folder.name)
            out_ann_dir = os.path.join(out_dir, 'object_detection')
            shutil.rmtree(out_ann_dir, ignore_errors=True)
            os.makedirs(out_ann_dir, exist_ok=True)

            metadata = set()
            shape = source_annotations[0].split('/')[-1].split('_')[1].split('x')
            width = int(shape[0])
            height = int(shape[1])
            for i, source_annotation in enumerate(tqdm(source_annotations)):
                boxes = []
                labels = []
                with open(source_annotation) as f:
                    for line in f.readlines():
                        elements = line.split()
                        center_x, center_y, w, h = list(map(float, elements[1:5]))
                        x1 = max(0, min(int((center_x - w / 2) * width), width - 1))
                        y1 = max(0, min(int((center_y - h / 2) * height), width - 1))
                        x2 = max(0, min(int((center_x + w / 2) * width), width - 1))
                        y2 = max(0, min(int((center_y + h / 2) * height), width - 1))
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(elements[0]))
                        metadata.add(int(elements[0]))

                filepath = os.path.join(out_ann_dir, filename_template % i)
                with open(filepath, 'w') as f:
                    for label, box in zip(labels, boxes):
                        f.write('%d %d %d %d %d\n' % (label, box[0], box[1], box[2], box[3]))

            metadata = list(metadata)
            metadata.sort()
            metadata_path = os.path.join(out_dir, 'metadata.txt')
            with open(metadata_path, 'w') as f:
                for class_id in metadata:
                    f.write('%d: %s\n' % (class_id, COCO_CLASS_DICT[class_id]))


def main():
    parser = argparse.ArgumentParser(description='SFU-HW-Objects dataset to dumps parser')
    parser.add_argument('--dataset-dir', dest='dataset_dir', type=str,
                        default="data/huawei/outputs/benchmark/SFU-HW-Objects-v2",
                        help="Path to video annotation file")
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default="data/huawei/outputs/benchmark/sfu_hw_objects_dataset",
                        help="Path to root of result dumps")
    args = parser.parse_args()

    convert_annotations(args.dataset_dir, args.output_dir)


if __name__ == '__main__':
    main()

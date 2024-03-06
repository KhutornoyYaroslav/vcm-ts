import os
import json
import shutil
import argparse
import cv2 as cv
from glob import glob
from subprocess import call

_YOLO_CLASS_ID_MAP = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorbike': 3,
    'bus': 5,
    'train': 6,
    'truck': 7
}

_COCO_CLASS_ID_MAP = {
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorbike': 4,
    'bus': 6,
    'train': 7,
    'truck': 8
}

_COLOR_MAP = {
    'person': (255, 0, 0),
    'bicycle': (0, 255, 0),
    'car': (0, 0, 255),
    'motorbike': (255, 255, 0),
    'bus': (0, 255, 255),
    'train': (255, 0, 255),
    'truck': (255, 255, 255)
}


def visualize_anno(video_path: str, result_root: str, color_map):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file '{video_path}'")
        return -1

    detector_filelist = sorted(glob(os.path.join(result_root, 'detector_yolo', '0000', "*.json")))
    # lpdetector_filelist = sorted(glob(os.path.join(result_root, 'lpdetector_yolo', '0000', "*.json")))
    lprdetector_filelist = sorted(glob(os.path.join(result_root, 'lprdetector_yolo', '0000', "*.json")))

    for detector_file, lprdetector_file in zip(detector_filelist, lprdetector_filelist):
        ret, frame = cap.read()
        if not ret:
            break

        with open(detector_file) as f:
            anno = json.load(f)
            for obj in anno['results']:
                x1, y1, w, h = obj['rect']
                color = color_map[obj['name']]
                cv.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        with open(lprdetector_file) as f:
            anno = json.load(f)
            for obj in anno['results']:
                # Liplate rect
                x1, y1, w, h = obj['rect']
                cv.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 2)
                # Liplate text
                if 'attribs' in obj:
                    for attr in obj['attribs']:
                        if 'data' in attr:
                            for d in attr['data']:
                                if 'label' in d:
                                    liplate_text = d['label']
                                    cv.putText(frame, liplate_text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                                               (255, 255, 255), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(0) & 0xFF == ord('q'):
            return 0


def convert_anno(path: str,
                 result_root: str,
                 video_filename: str,
                 yolo_class_id_map: dict,
                 filename_template="im%05d.txt"):
    # Object detection
    res_folder_detector_yolo = os.path.join(result_root, video_filename, 'annotations')
    shutil.rmtree(res_folder_detector_yolo, ignore_errors=True)
    os.makedirs(res_folder_detector_yolo, exist_ok=True)

    # Convert annotation
    with open(path, 'r') as f:
        data = json.load(f)

    obj_class_map = {}
    for obj in data['objects']:
        obj_class_map[obj['key']] = obj['classTitle']

    for frame_data in data['frames']:
        frame_id = frame_data['index'] + 1

        boxes = []
        labels = []
        for figure_data in frame_data['figures']:
            class_name = obj_class_map[figure_data['objectKey']]
            [x1, y1], [x2, y2] = figure_data['geometry']['points']['exterior']

            boxes.append([x1, y1, x2, y2])
            labels.append(yolo_class_id_map[class_name])

        # Object detection
        filepath = os.path.join(res_folder_detector_yolo, filename_template % frame_id)

        with open(filepath, 'w') as f:
            for label, box in zip(labels, boxes):
                f.write('%d %d %d %d %d\n' % (label, box[0], box[1], box[2], box[3]))


def video_to_images(video_path: str, out_path: str):
    call([
        'ffmpeg',
        '-i', video_path,
        out_path
    ])


def main():
    parser = argparse.ArgumentParser(description='SVS Codec Testing')
    parser.add_argument('--anno-path', dest='anno_path', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests/ds0/ann/test_0_short.mp4.json",
                        help="Path to video annotation file")
    parser.add_argument('--result-root', dest='result_root', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests_dumps/test_0_short_dumps",
                        help="Path to root of result dumps")
    parser.add_argument('--video-path', dest='video_path', type=str,
                        default="/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/test_0_short.mp4",
                        help="Path video")
    args = parser.parse_args()

    anno_paths = [
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests/ds0/ann/test_0_short.mp4.json",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests/ds0/ann/test_10_short.mp4.json",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests/ds0/ann/test_18_short.mp4.json",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/huawei_tests/ds0/ann/test_21_short.mp4.json"
    ]
    out_path = "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark"
    video_paths = [
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/test_0_short.mp4",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/test_10_short.mp4",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/test_18_short.mp4",
        "/home/alexnevskiy/PycharmProjects/vcm-ts/data/huawei/outputs/benchmark/test_21_short.mp4"
    ]

    for anno_path, video_path in zip(anno_paths, video_paths):
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        convert_anno(anno_path, out_path, video_filename, _COCO_CLASS_ID_MAP)

        images_path = os.path.join(out_path, video_filename, 'images')
        shutil.rmtree(images_path, ignore_errors=True)
        os.makedirs(images_path, exist_ok=True)
        images_path = os.path.join(images_path, "im%05d.png")
        video_to_images(video_path, images_path)

    # convert_anno(args.anno_path, args.result_root, _COCO_CLASS_ID_MAP)
    # visualize_anno(args.video_path, args.result_root, _COLOR_MAP)


if __name__ == '__main__':
    main()

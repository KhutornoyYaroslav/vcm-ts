import os
import json
import shutil
import argparse
import cv2 as cv
from glob import glob


_YOLO_CLASS_ID_MAP = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorbike': 3,
    'bus': 5,
    'train': 6,
    'truck' : 7
}


_COLOR_MAP = {
    'person': (255, 0, 0),
    'bicycle': (0, 255, 0),
    'car': (0, 0, 255),
    'motorbike': (255, 255, 0),
    'bus': (0, 255, 255),
    'train': (255, 0, 255),
    'truck' : (255, 255, 255)
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
                cv.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        with open(lprdetector_file) as f:
            anno = json.load(f)
            for obj in anno['results']:
                # Liplate rect
                x1, y1, w, h = obj['rect']
                cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
                # Liplate text
                if 'attribs' in obj:
                    for attr in obj['attribs']:
                        if 'data' in attr:
                            for d in attr['data']:
                                if 'label' in d:
                                    liplate_text = d['label']
                                    cv.putText(frame, liplate_text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(0) & 0xFF == ord('q'):
            return 0
        

def convert_anno(path: str,
                 result_root: str,
                 yolo_class_id_map: dict,
                 filename_template = "frame_%08d.json"):
    # Object detection
    res_folder_detector_yolo = os.path.join(result_root, 'detector_yolo', '0000')
    shutil.rmtree(res_folder_detector_yolo, ignore_errors=True)
    os.makedirs(res_folder_detector_yolo, exist_ok=True)

    # License plate detection
    res_folder_lpdetector_yolo = os.path.join(result_root, 'lpdetector_yolo', '0000')
    shutil.rmtree(res_folder_lpdetector_yolo, ignore_errors=True)
    os.makedirs(res_folder_lpdetector_yolo, exist_ok=True)

    # License plate detection + recognition
    res_folder_lprdetector_yolo = os.path.join(result_root, 'lprdetector_yolo', '0000')
    shutil.rmtree(res_folder_lprdetector_yolo, ignore_errors=True)
    os.makedirs(res_folder_lprdetector_yolo, exist_ok=True)

    # Convert annotation
    with open(path, 'r') as f:
        data = json.load(f)

    frame_w, frame_h = data['size']['width'], data['size']['height']
    
    obj_class_map = {}
    for obj in data['objects']:
        obj_class_map[obj['key']] = obj['classTitle']

    for frame_data in data['frames']:
        frame_id = frame_data['index']

        detector_yolo_results = []
        lpdetector_yolo_results = []
        lprdetector_yolo_results = []
        for figure_data in frame_data['figures']:
            class_name = obj_class_map[figure_data['objectKey']]
            [x1, y1], [x2, y2] = figure_data['geometry']['points']['exterior']

            if class_name == "liplate":
                # License plate detection
                obj_info = {
                    'prob': 0.99,
                    'rect': [x1, y1, x2 - x1, y2 - y1]
                }
                lpdetector_yolo_results.append(obj_info)

                # License plate detection + recognition
                obj_tags = []
                for obj in data['objects']:
                    if obj['key'] == figure_data['objectKey']:
                        obj_tags = obj['tags']

                liplate_text = None
                for tag in obj_tags:
                    if tag['name'] == "text":
                        liplate_text = tag['value']
                        break

                if liplate_text != None:
                    obj_info = {
                        'prob': 0.99,
                        'rect': [x1, y1, x2 - x1, y2 - y1],
                        'attribs': [
                            {
                                'name': "lpr",
                                'type': "CRNNLPRec",
                                'data': [
                                    {
                                        'label': liplate_text
                                    }
                                ]
                            }
                        ]
                    }
                    lprdetector_yolo_results.append(obj_info)
            else:
                # Object detection
                obj_info = {
                    'klass': yolo_class_id_map[class_name],
                    'name': class_name,
                    'rect': [x1, y1, x2 - x1, y2 - y1],
                    'score': 0.99,
                }
                detector_yolo_results.append(obj_info)

        # Object detection
        detector_yolo_data = {
            'id': frame_id + 1,
            'framesize': [frame_w, frame_h],
            'name': 'detector_yolo',
            'type': 'YOLOv4',
            'results': detector_yolo_results
        }

        # License plate detection
        lpdetector_yolo_data = {
            'id': frame_id + 1,
            'framesize': [frame_w, frame_h],
            'name': 'lpdetector_yolo',
            'type': 'YOLOv8LP',
            'results': lpdetector_yolo_results
        }

        # License plate detection + recognition
        lprdetector_yolo_data = {
            'id': frame_id + 1,
            'framesize': [frame_w, frame_h],
            'name': 'lprdetector_yolo',
            'type': 'ROIProxy',
            'results': lprdetector_yolo_results
        }

        # Object detection
        filepath = os.path.join(res_folder_detector_yolo, filename_template % frame_id)
        with open(filepath, 'x') as f:
            json.dump(detector_yolo_data, f)

        # License plate detection
        filepath = os.path.join(res_folder_lpdetector_yolo, filename_template % frame_id)
        with open(filepath, 'x') as f:
            json.dump(lpdetector_yolo_data, f)

        # License plate detection + recognition
        filepath = os.path.join(res_folder_lprdetector_yolo, filename_template % frame_id)
        with open(filepath, 'x') as f:
            json.dump(lprdetector_yolo_data, f)


def main():
    parser = argparse.ArgumentParser(description='SVS Codec Testing')
    parser.add_argument('--anno-path', dest='anno_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/test_videos/huawei_liplates/ds0/ann/test_14_liplates.mp4.json",
                        help="Path to video annotation file")
    parser.add_argument('--result-root', dest='result_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/test_videos/huawei_liplates_dumps/test_14_liplates_dumps",
                        help="Path to root of result dumps")
    parser.add_argument('--video-path', dest='video_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/test_videos/test_14_liplates.mp4",
                        help="Path video")
    args = parser.parse_args()

    convert_anno(args.anno_path, args.result_root, _YOLO_CLASS_ID_MAP)
    visualize_anno(args.video_path, args.result_root, _COLOR_MAP)


if __name__ == '__main__':
    main()

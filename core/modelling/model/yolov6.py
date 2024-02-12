import os
import torch
import torchvision


_YOLOv6_WEIGHTS_DIR = './YOLOv6/weights'


class YOLOv6Detector:
    _class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, model_type: str = 'yolov6l', device: str = 'cuda'):
        # Download model weights from v0.4.0 release instead of v0.3.0 as by default in YOLOv6/hubconf.py
        if not os.path.exists(_YOLOv6_WEIGHTS_DIR):
            os.mkdir(_YOLOv6_WEIGHTS_DIR)

        weights_filepath = os.path.join(_YOLOv6_WEIGHTS_DIR, f'{model_type}.pt')
        if not os.path.exists(weights_filepath):
            torch.hub.download_url_to_file(f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{model_type}.pt", weights_filepath)

        # Load model
        self.detector = torch.hub.load('YOLOv6', model_type, source='local')
        self.detector.model.to(device)
        self.detector.model.half()

    def detect(self, tensor, to_cpu=True, to_list=False):
        with torch.no_grad():
            raw_detection = self.detector.model(tensor.half())

        direct_detection = raw_detection[0]
        suppressed_detection = self._non_max_suppression(direct_detection)
        if to_cpu:
            suppressed_detection = [sd.cpu() for sd in suppressed_detection]
        if to_list:
            suppressed_detection = [sd.tolist() for sd in suppressed_detection]
        return suppressed_detection

    @staticmethod
    def filter_bboxes_by_class(bboxes, allowed_classes):
        filtered_bboxes = []
        for bbox in bboxes:
            if YOLOv6Detector._class_names[int(bbox[5])] in allowed_classes:
                filtered_bboxes.append(bbox)
        return filtered_bboxes
    
    @staticmethod
    def filter_bboxes_by_score(bboxes, threshold: float):
        assert 0 <= threshold <= 1.0

        filtered_bboxes = []
        for bbox in bboxes:
            if bbox[4] >= threshold:
                filtered_bboxes.append(bbox)
        return filtered_bboxes
    
    @staticmethod
    def get_bboxes_coordinates(bboxes, max_width: None, max_height: None, padding: int = 0):
        bboxes_coords = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[0:4]
            x1 = x1 - padding
            y1 = y1 - padding
            x2 = x2 + padding
            y2 = y2 + padding
            if max_width != None:
                x1 = max(min(x1, max_width), 0)
                x2 = max(min(x2, max_width), 0)
            if max_height != None:
                y1 = max(min(y1, max_height), 0)
                y2 = max(min(y2, max_height), 0)
            bboxes_coords.append([int(x1), int(y1), int(x2), int(y2)])

        return bboxes_coords

    
    # cut version of non_max_suppression from yolov6
    def _non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                             multi_label=False, max_det=300):
        num_classes = prediction.shape[2] - 5  # number of classes
        pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres,
                                            torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates

        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        multi_label &= num_classes > 1  # multiple labels per box

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            def xywh2xyxy(temp_x):
                temp_y = temp_x.clone()
                temp_y[:, 0] = temp_x[:, 0] - temp_x[:, 2] / 2  # top left x
                temp_y[:, 1] = temp_x[:, 1] - temp_x[:, 3] / 2  # top left y
                temp_y[:, 2] = temp_x[:, 0] + temp_x[:, 2] / 2  # bottom right x
                temp_y[:, 3] = temp_x[:, 1] + temp_x[:, 3] / 2  # bottom right y
                return temp_y

            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
            else:  # Only keep the class with highest scores.
                conf, class_idx = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]

        return output

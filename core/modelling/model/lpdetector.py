import torch
from core.utils.padding import pad_image
from Automatic_Number_Plate_Detection_Recognition_YOLOv8.ultralytics.yolo.utils import ops
from Automatic_Number_Plate_Detection_Recognition_YOLOv8.ultralytics.nn.autobackend import AutoBackend
from Automatic_Number_Plate_Detection_Recognition_YOLOv8.ultralytics.yolo.engine.predictor import BasePredictor


class LiplateDetectionPredictor(BasePredictor): # TODO: to models
    def preprocess(self, img):
        img = pad_image(img, 32)
        img = torch.from_numpy(img).to(self.model.device)
        img = img.permute(2, 0, 1).unsqueeze(0) # H, W, C -> N, C, H, W
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        return preds

    def setup(self, weights, device):
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = AutoBackend(weights, device=device, dnn=self.args.dnn, fp16=self.args.half)
        model.eval()

        self.model = model
        self.done_setup = True
        self.device = device

        return model

    def predict(self, img):
        im = self.preprocess(img)
        preds = self.model.forward(im, augment=False, visualize=False)
        preds = self.postprocess(preds)

        return preds

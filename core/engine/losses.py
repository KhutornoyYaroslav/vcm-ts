import math
import torch
import torchvision
import numpy as np
from torch import nn
from core.utils.ssim import MS_SSIM
from core.modelling.model.yolov6 import YOLOv6Detector
from scipy.optimize import linear_sum_assignment
from torchvision.models.feature_extraction import create_feature_extractor


class RateLoss(nn.Module):
    def __init__(self, reduction_dims: list = [], log_base=2):
        super(RateLoss, self).__init__()
        self.log_base = log_base
        self.reduction_dims = reduction_dims

    def forward(self, likelihoods, masks = None):
        rate = torch.log(likelihoods) / -math.log(self.log_base) # (N, T, C, H, W)
        if masks != None:
            rate = rate * masks # (N, T, C, H, W) * (N, T, 1, H, W)
        rate_total = torch.sum(rate, dim=self.reduction_dims)

        return rate_total


class MSELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(MSELoss, self).__init__()
        self.MSE = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, labels, masks = None):
        results = self.MSE(inputs, labels)
        if masks != None:
            results = results * masks

        return results


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, method='ms-ssim'):
        super(SSIMLoss, self).__init__()

        if method == 'ms-ssim':
            self.metric = MS_SSIM(data_range=data_range, size_average=False)
        else:
            raise ValueError("Invalid argument 'method={0}'".format(method))

    def forward(self, inputs, labels): # (N, T, C, H, W)
        metrics = []
        for input, label in zip(inputs, labels): # Iterate over batch (N) dimension
            metric = self.metric(input, label)
            metrics.append(metric)
        metrics = torch.stack(metrics, dim=0)

        return metrics


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, masks = None):
        results = torch.sqrt((pred - target)**2 + self.eps)
        if masks is not None:
            results = results * masks

        return results


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class FasterRCNNPerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super(FasterRCNNPerceptualLoss, self).__init__()
        # Create model
        self.pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=self.pretrained_weights)
        self.model.eval()
        self.to(device)
        # Get features
        self.return_nodes = {
            'body.layer1': 'x4',
            'body.layer2': 'x8',
            'body.layer3': 'x16',
            'body.layer4': 'x32',
        }
        self.features = create_feature_extractor(self.model.backbone, return_nodes=self.return_nodes)

    def forward(self, input, target, feature_layers=['x4', 'x8', 'x16', 'x32']):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Calculate features
        with torch.no_grad():
            f_input = self.features.forward(input)
            f_target = self.features.forward(target)

        # Calculate loss
        loss = 0.0
        for key in f_input.keys():
            if key in feature_layers:
                loss += torch.nn.functional.l1_loss(f_input[key], f_target[key])

        return torch.FloatTensor([loss])


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.roi_model = YOLOv6Detector("yolov6l.pt")

    def forward(self, inputs, targets, targets_are_images=True):
        if targets_are_images:
            bboxes = self.roi_model.detect(targets)

        new_bboxes = self.roi_model.detect(inputs)

        loss = 0
        for bbx1, bbx2 in zip(bboxes, new_bboxes):  # batches
            loss += self._arbitrary_size_iou_loss(bbx1, bbx2)
        loss /= len(bboxes)

        return loss

    @staticmethod
    def _iou(box1, box2):
        if int(box1[5]) != int(box2[5]):  # different classes
            return 0

        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union

    def _arbitrary_size_iou_loss(self, bboxes1, bboxes2):
        # TODO: include scores
        # cp_bboxes1 = bboxes1.detach().clone()
        bboxes1 = bboxes1[bboxes1[:, 4] > 0.001]
        bboxes1 = bboxes1[np.isin(bboxes1[:, 5].int(), self.roi_model.allowed_class_idxs)]
        bboxes2 = bboxes2[bboxes2[:, 4] > 0.001]
        bboxes2 = bboxes2[np.isin(bboxes2[:, 5].int(), self.roi_model.allowed_class_idxs)]
        num_boxes_a = len(bboxes1)
        num_boxes_b = len(bboxes2)
        if num_boxes_a == 0 and num_boxes_b == 0:  # both empty, good
            return np.float64(0)
        if (num_boxes_a == 0) != (num_boxes_b == 0):  # only one empty (xor), bad
            return np.float64(1)

        # Calculate the IoU matrix
        iou_matrix = np.zeros((num_boxes_a, num_boxes_b))
        for i in range(num_boxes_a):
            for j in range(num_boxes_b):
                iou_matrix[i, j] = self._iou(bboxes1[i], bboxes2[j])

        # Use the Hungarian algorithm to find the optimal matching of boxes
        row_ind, col_ind = linear_sum_assignment(-1 * iou_matrix)

        # Calculate the overall IoU
        ious = 0
        for i in range(len(row_ind)):
            ious += iou_matrix[row_ind[i], col_ind[i]]

        n_pairs = len(row_ind)
        n_instances = len(row_ind) + (num_boxes_a - n_pairs) + (num_boxes_b - n_pairs)
        ious /= n_instances

        iou_loss = 1 - ious
        return iou_loss

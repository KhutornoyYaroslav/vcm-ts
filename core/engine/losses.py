import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from ultralytics import YOLO

from DCVC_HEM.src.utils.stream_helper import get_padding_size


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load('pretrained/vgg16-397923af.pth'))
        blocks.append(vgg16.features[:4].eval())
        blocks.append(vgg16.features[4:9].eval())
        blocks.append(vgg16.features[9:16].eval())
        blocks.append(vgg16.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
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
                loss_ = torch.nn.functional.mse_loss(x, y, reduction='none')
                loss += torch.mean(loss_, dim=(1, 2, 3))
        return loss  # (N)


class FasterRCNNPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(FasterRCNNPerceptualLoss, self).__init__()
        # Create model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        self.model.load_state_dict(torch.load('pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        # Get features
        self.return_nodes = {
            'body.layer1': 'x4',
            'body.layer2': 'x8',
            'body.layer3': 'x16',
            'body.layer4': 'x32',
        }
        self.features = create_feature_extractor(self.model.backbone, return_nodes=self.return_nodes)
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, input, target, feature_layers=['x4', 'x8', 'x16', 'x32']):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Calculate features
        f_input = self.features.forward(input)
        f_target = self.features.forward(target)

        # Calculate loss
        loss = 0.0
        for key in f_input.keys():
            if key in feature_layers:
                loss_ = torch.nn.functional.mse_loss(f_input[key], f_target[key], reduction='none')
                loss += torch.mean(loss_, dim=(1, 2, 3))

        return loss


class FasterRCNNFPNPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(FasterRCNNFPNPerceptualLoss, self).__init__()
        # Create model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        self.model.load_state_dict(torch.load('pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        # Get features
        self.features = self.model.backbone
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, input, target, feature_layers=['0', '1', '2', '3', 'pool']):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Calculate features
        f_input = self.features.forward(input)
        f_target = self.features.forward(target)

        # Calculate loss
        loss = 0.0
        for key in f_input.keys():
            if key in feature_layers:
                loss_ = torch.nn.functional.mse_loss(f_input[key], f_target[key], reduction='none')
                loss += torch.mean(loss_, dim=(1, 2, 3))

        return loss


class YOLOV8PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(YOLOV8PerceptualLoss, self).__init__()
        # Create model
        self.model = YOLO('pretrained/yolov8m.pt')
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.model.model.eval()

    def get_features(self, input):
        y = []
        features = None
        for m in self.model.model.model:
            if m.f != -1:  # if not from previous layer
                input = y[m.f] if isinstance(m.f, int) else [input if j == -1 else y[j] for j in
                                                             m.f]  # from earlier layers
            if torch.is_tensor(input):
                features = input  # keep the last tensor as features
            input = m(input)  # run
            if torch.is_tensor(input):
                features = input  # keep the last tensor as features
            y.append(input if m.i in self.model.model.save else None)  # save output
        if torch.is_tensor(input):
            features = input  # keep the last tensor as features
        return features

    def forward(self, input, target):
        pic_height = input.shape[2]
        pic_width = input.shape[3]
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, p=32)
        input_padded = torch.nn.functional.pad(
            input,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        target_padded = torch.nn.functional.pad(
            target,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )
        f_input = self.get_features(input_padded)
        f_target = self.get_features(target_padded)

        # Calculate loss
        loss_ = torch.nn.functional.mse_loss(f_input, f_target, reduction='none')
        loss = torch.mean(loss_, dim=(1, 2, 3))

        return loss

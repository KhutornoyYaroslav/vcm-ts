import torch
import torchvision
from ultralytics import YOLO

from DCVC_HEM.src.utils.stream_helper import get_padding_size


class FasterRCNNResNetPerceptualLoss(torch.nn.Module):
    def __init__(self, requires_grad: bool = False):
        super(FasterRCNNResNetPerceptualLoss, self).__init__()
        # initialize model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        model.load_state_dict(torch.load('pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
        model_body = model.backbone.body
        # create feature slices
        self.slice1 = torch.nn.Sequential(
            model_body.conv1,
            model_body.bn1,
            model_body.relu
        )
        self.slice2 = torch.nn.Sequential(
            model_body.maxpool,
            model_body.layer1
        )
        self.slice3 = model_body.layer2
        self.slice4 = model_body.layer3
        self.slice5 = model_body.layer4
        # disable grads
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        # norm
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward_features_(self, x):
        f = self.slice1(x)
        f_out1 = f
        f = self.slice2(f)
        f_out2 = f
        f = self.slice3(f)
        f_out3 = f
        f = self.slice4(f)
        f_out4 = f
        f = self.slice5(f)
        f_out5 = f

        return {
            '1': f_out1,
            '2': f_out2,
            '3': f_out3,
            '4': f_out4,
            '5': f_out5
        }

    @staticmethod
    def normalize_features(in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)

    def disable_gradients(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target, normalize: bool = True, resize: bool = True,
                feature_layers=['1', '2', '3', '4', '5']):
        # check shape
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # clump
        input = input.clamp(0, 1)
        target = target.clamp(0, 1)

        # transforms
        if normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        if resize:
            input = torch.nn.functional.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = torch.nn.functional.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        # get features
        fs_input = self.forward_features_(input)
        fs_target = self.forward_features_(target)

        # calc loss
        loss = []
        for key in fs_input.keys():
            if key in feature_layers:
                f_input_norm = self.normalize_features(fs_input[key])
                f_target_norm = self.normalize_features(fs_target[key])
                loss_ = torch.nn.functional.mse_loss(f_input_norm, f_target_norm, reduction='none')
                loss_ = torch.mean(loss_, dim=(1, 2, 3))
                loss.append(loss_)

        loss = torch.stack(loss)
        loss = torch.sum(loss, 0)

        return loss


class FasterRCNNFPNPerceptualLoss(torch.nn.Module):
    def __init__(self, requires_grad: bool = False):
        super(FasterRCNNFPNPerceptualLoss, self).__init__()
        # initialize model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        model.load_state_dict(torch.load('pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
        # get features
        self.features = model.backbone
        # disable grads
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        # norm
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def normalize_features(in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)

    def disable_gradients(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target, normalize: bool = True, resize: bool = True,
                feature_layers=['0', '1', '2', '3', 'pool']):
        # check shape
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # clump
        input = input.clamp(0, 1)
        target = target.clamp(0, 1)

        # transforms
        if normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        if resize:
            input = torch.nn.functional.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = torch.nn.functional.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        # get features
        fs_input = self.features.forward(input)
        fs_target = self.features.forward(target)

        # calc loss
        loss = []
        for key in fs_input.keys():
            if key in feature_layers:
                f_input_norm = self.normalize_features(fs_input[key])
                f_target_norm = self.normalize_features(fs_target[key])
                loss_ = torch.nn.functional.mse_loss(f_input_norm, f_target_norm, reduction='none')
                loss_ = torch.mean(loss_, dim=(1, 2, 3))
                loss.append(loss_)

        loss = torch.stack(loss)
        loss = torch.sum(loss, 0)

        return loss


class YOLOV8PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(YOLOV8PerceptualLoss, self).__init__()
        # Create model
        self.model = YOLO('pretrained/yolov8m.pt')
        self.model = self.model.model
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.model.eval()

    def disable_gradients(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_features(self, input, layers=[0, 1, 3, 5, 7, 15, 18, 21]):
        y = []
        features = []
        for i, m in enumerate(self.model.model):
            if m.f != -1:  # if not from previous layer
                input = y[m.f] if isinstance(m.f, int) else [input if j == -1 else y[j] for j in
                                                             m.f]  # from earlier layers
            input = m(input)  # run
            if i in layers:
                features.append(input)
            y.append(input if m.i in self.model.save else None)  # save output
            if len(features) == len(layers):
                break
        return {
            '1': features[0],
            '2': features[1],
            '3': features[2],
            '4': features[3],
            '5': features[4],
            '3_deep': features[5],
            '4_deep': features[6],
            '5_deep': features[7]
        }

    def forward(self, target, input, feature_layers=['1', '2', '3', '4', '5', '3_deep', '4_deep', '5_deep']):
        input = input.clamp(0, 1)
        target = target.clamp(0, 1)

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
        fs_input = self.get_features(input_padded)
        fs_target = self.get_features(target_padded)

        # Calculate loss
        loss = []
        for key in fs_input.keys():
            if key in feature_layers:
                loss_ = torch.nn.functional.mse_loss(fs_input[key], fs_target[key], reduction='none')
                loss_ = torch.mean(loss_, dim=(1, 2, 3))
                loss.append(loss_)

        loss = torch.stack(loss)
        loss = torch.sum(loss, 0)

        return loss

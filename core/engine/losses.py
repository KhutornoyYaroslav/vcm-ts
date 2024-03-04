import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


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

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
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
                loss_ = torch.nn.functional.l1_loss(x, y, reduction='none')
                loss += torch.mean(loss_, dim=(1, 2, 3))
        return loss # (N)


class FasterRCNNPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(FasterRCNNPerceptualLoss, self).__init__()
        # Create model
        self.pretrained_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=self.pretrained_weights)
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
        # TODO: reduce mean only by (1, 2, 3) dimensions
        loss = 0.0
        for key in f_input.keys():
            if key in feature_layers:
                loss += torch.nn.functional.l1_loss(f_input[key], f_target[key])

        return loss

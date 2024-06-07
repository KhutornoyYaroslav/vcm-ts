import cv2 as cv
import numpy as np
import torch

from .functional import make_array_divisible_by


class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, targets, masks=None, resids=None):
        for t in self.transforms:
            inputs, targets, masks, resids = t(inputs, targets, masks, resids)
        return inputs, targets, 0 if masks is None else masks, 0 if resids is None else resids


class ConvertFromInts:
    def __call__(self, inputs, targets, masks=None, resids=None):
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        if masks is not None:
            masks = masks.astype(np.float32)
        if resids is not None:
            resids = resids.astype(np.float32)
        return inputs, targets, masks, resids


class Clip(object):
    def __init__(self, min: float = 0.0, max: float = 255.0):
        self.min = min
        self.max = max
        assert self.max >= self.min, "min val must be >= max val"

    def __call__(self, inputs, targets, masks=None, resids=None):
        inputs = np.clip(inputs, self.min, self.max)
        targets = np.clip(targets, self.min, self.max)
        return inputs, targets, masks, resids


class Normalize(object):
    def __init__(self, norm_mask: bool = True, norm_resids: bool = True):
        self.norm_mask = norm_mask
        self.norm_resids = norm_resids

    def __call__(self, inputs, targets, masks=None, resids=None):
        inputs = inputs.astype(np.float32) / 255.0
        targets = targets.astype(np.float32) / 255.0
        if masks is not None and self.norm_mask:
            masks = masks.astype(np.float32) / 255.0
        if resids is not None and self.norm_resids:
            resids = resids.astype(np.float32) / 255.0
        return inputs, targets, masks, resids


class ToTensor:
    def __call__(self, inputs, targets, masks=None, resids=None):
        # (T, H, W, C) -> (T, C, H, W)
        inputs = torch.from_numpy(inputs.astype(np.float32)).permute(0, 3, 1, 2)
        targets = torch.from_numpy(targets.astype(np.float32)).permute(0, 3, 1, 2)
        if masks is not None:
            masks = torch.from_numpy(masks.astype(np.float32)).permute(0, 3, 1, 2)
        if resids is not None:
            resids = torch.from_numpy(resids.astype(np.float32)).permute(0, 3, 1, 2)
        return inputs, targets, masks, resids


class MakeDivisibleBy:
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, inputs, targets, masks=None, resids=None):
        inputs = make_array_divisible_by(inputs, self.factor)
        targets = make_array_divisible_by(targets, self.factor)

        if masks is not None:
            masks = make_array_divisible_by(masks, self.factor)

        if resids is not None:
            resids = make_array_divisible_by(resids, self.factor)

        return inputs, targets, masks, resids


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, inputs, targets, masks=None, resids=None):
        # (T, H, W, C)
        if self.current == 'BGR' and self.transform == 'RGB':
            for i, _ in enumerate(inputs):
                inputs[i] = cv.cvtColor(inputs[i], cv.COLOR_BGR2RGB)
            for i, _ in enumerate(targets):
                targets[i] = cv.cvtColor(targets[i], cv.COLOR_BGR2RGB)
        elif self.current == 'RGB' and self.transform == 'BGR':
            for i, _ in enumerate(inputs):
                inputs[i] = cv.cvtColor(inputs[i], cv.COLOR_RGB2BGR)
            for i, _ in enumerate(targets):
                targets[i] = cv.cvtColor(targets[i], cv.COLOR_RGB2BGR)
        else:
            raise NotImplementedError

        return inputs, targets, masks, resids


class RandomCrop(object):
    def __init__(self, w: int, h: int, probabilty: float = 0.5):
        assert w > 0 and h > 0, "Width and height of crop area must be positive"
        self.crop_w = w
        self.crop_h = h
        self.p = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, inputs, targets, masks=None, resids=None):
        if np.random.choice([0, 1], size=1, p=[1 - self.p, self.p]):
            _, h, w, _ = inputs.shape
            crop_x = int(np.random.random() * (w - self.crop_w))
            crop_y = int(np.random.random() * (h - self.crop_h))
            assert crop_x >= 0 and crop_y >= 0, "Image size must not be smaller than crop size"

            inputs = inputs[:, crop_y:crop_y + self.crop_h, crop_x:crop_x + self.crop_w, :]
            targets = targets[:, crop_y:crop_y + self.crop_h, crop_x:crop_x + self.crop_w, :]
            assert masks is None, "Cropping for masks was not implemented"
            assert resids is None, "Cropping for resids was not implemented"

        return inputs, targets, masks, resids


class CentralCrop(object):
    def __init__(self, w: int, h: int, probabilty: float = 0.5):
        assert w > 0 and h > 0, "Width and height of crop area must be positive"
        self.crop_w = w
        self.crop_h = h
        self.p = np.clip(probabilty, 0.0, 1.0)

    def __call__(self, inputs, targets, masks=None, resids=None):
        if np.random.choice([0, 1], size=1, p=[1 - self.p, self.p]):
            _, h, w, _ = inputs.shape
            crop_x = int((w - self.crop_w) / 2)
            crop_y = int((h - self.crop_h) / 2)
            assert crop_x >= 0 and crop_y >= 0, "Image size must not be smaller than crop size"

            inputs = inputs[:, crop_y:crop_y + self.crop_h, crop_x:crop_x + self.crop_w, :]
            targets = targets[:, crop_y:crop_y + self.crop_h, crop_x:crop_x + self.crop_w, :]
            assert masks is None, "Cropping for masks was not implemented"
            assert resids is None, "Cropping for resids was not implemented"

        return inputs, targets, masks, resids

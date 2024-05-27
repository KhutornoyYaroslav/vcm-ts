from glob import glob

import cv2 as cv
import numpy
import torch
from torch.utils.data import Dataset

from ..transforms.transforms import (
    ConvertFromInts,
    Clip,
    Normalize,
    ToTensor,
    TransformCompose,
    ConvertColor,
    MakeDivisibleBy
)


class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, cfg):
        self.cfg = cfg
        self.root_dir = root_dir
        self.divisible_by = cfg.INPUT.MAKE_DIVISIBLE_BY
        self.image_infos = self.read_images(self.root_dir)
        self.transforms = self.build_transforms(self.divisible_by, True)

    def __len__(self):
        return len(self.image_infos)

    def read_images(self, root: str):
        images = sorted(glob(root + "/images/*"))
        annotations = sorted(glob(root + "/object_detection/*"))
        assert len(images) == len(annotations)

        image_infos = []
        for image, annotation in zip(images, annotations):
            image_infos.append({"image": image, "annotation": annotation})

        return image_infos

    def build_transforms(self, div_by: int = 1, to_tensor: bool = True):
        transform = [
            MakeDivisibleBy(div_by),
            ConvertColor('BGR', 'RGB'),
            ConvertFromInts(),
            Clip()
        ]

        if to_tensor:
            transform = transform + [Normalize(False, False), ToTensor()]

        transform = TransformCompose(transform)
        return transform

    def read_object_detection(self, annotation_path):
        boxes = []
        labels = []
        with open(annotation_path) as f:
            for line in f.readlines():
                elements = list(map(int, line.split()))
                boxes.append(elements[1:5])
                labels.append(elements[0])
        target = dict(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.int64)
        )
        return target

    def __getitem__(self, idx):
        # Scan files
        image_info_path = self.image_infos[idx]
        image_path = image_info_path["image"]
        annotation_path = image_info_path["annotation"]

        # Read image
        image = cv.imread(image_path)
        image = numpy.expand_dims(image, axis=0)  # (T, H, W, C)
        image_copy = image.copy()
        annotation = self.read_object_detection(annotation_path)

        # Apply transforms
        if self.transforms:
            image, _, _, _ = self.transforms(image, image_copy)

        return image, annotation  # (T, C, H, W)

import os
from glob import glob
from typing import Tuple

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from ..transforms.transforms import (
    ConvertFromInts,
    Clip,
    Normalize,
    ToTensor,
    TransformCompose,
    ConvertColor,
    MakeDivisibleBy,
    RandomCrop,
    CentralCrop
)


class SequenceDataset(Dataset):
    def __init__(self, root_dir, cfg, dir_list: str = '', is_train: bool = True, to_tensor: bool = True,
                 print_warn: bool = False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.dir_list = dir_list
        self.divisible_by = cfg.INPUT.MAKE_DIVISIBLE_BY
        self.inputs_dirname_template = cfg.DATASET.SUBDIR_INPUTS
        self.seq_length = cfg.DATASET.SEQUENCE_LENGTH
        self.seq_stride = cfg.DATASET.SEQUENCE_STRIDE
        self.sequences = self.read_sequences(self.root_dir, self.dir_list, self.seq_length * self.seq_stride,
                                             print_warn)
        self.transforms = self.build_transforms(cfg.INPUT.IMAGE_SIZE, self.divisible_by, is_train, to_tensor)

    def __len__(self):
        return len(self.sequences)

    def read_sequences(self, root: str, dir_list: str, min_length: int, print_warn: bool = False):
        if dir_list == '':
            seqs = sorted(glob(root + "/*/*"))
        else:
            with open(dir_list, 'r') as f:
                seqs = []
                for line in f:
                    subdir = line.strip()
                    if len(subdir):
                        seqs.append(root + "/" + line.strip())

        seqs_filtered = []
        for s in seqs:
            inputs_len = len(glob(os.path.join(s, self.inputs_dirname_template, "*")))

            if inputs_len >= min_length:
                seqs_filtered.append(s)
            else:
                if print_warn:
                    print(f"Skip sequence due to length: '{s}'")

        return seqs_filtered

    def build_transforms(self, img_size: Tuple[int, int], div_by: int = 1, is_train: bool = True,
                         to_tensor: bool = True):
        if is_train:
            transform = [
                RandomCrop(img_size[0], img_size[1], 1.0),
                MakeDivisibleBy(div_by),
                ConvertColor('BGR', 'RGB'),
                ConvertFromInts(),
                Clip()
            ]
        else:
            transform = [
                CentralCrop(img_size[0], img_size[1], 1.0),
                MakeDivisibleBy(div_by),
                ConvertColor('BGR', 'RGB'),
                ConvertFromInts(),
                Clip()
            ]

        if to_tensor:
            transform = transform + [Normalize(False, False), ToTensor()]

        transform = TransformCompose(transform)
        return transform

    def __getitem__(self, idx):
        # Scan files
        seq_path = self.sequences[idx]
        input_paths = sorted(glob(os.path.join(seq_path, self.inputs_dirname_template, "*")))

        # Reduce sequence length
        input_paths = input_paths[:self.seq_stride * self.seq_length:self.seq_stride]

        # Read images
        inputs = []
        for i in range(len(input_paths)):
            input = cv.imread(input_paths[i])
            inputs.append(input)

        input_seq = np.stack(inputs, axis=0)  # (T, H, W, C)
        target_seq = input_seq.copy()

        # Apply transforms
        if self.transforms:
            input_seq, target_seq, _, _ = self.transforms(input_seq, target_seq)

        return input_seq, target_seq  # (T, C, H, W)

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            input_seq, target_seq = self.__getitem__(i)

            for input, target in zip(input_seq, target_seq):
                input = (input.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                target = (target.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

                input = cv.cvtColor(input, cv.COLOR_RGB2BGR)
                target = cv.cvtColor(target, cv.COLOR_RGB2BGR)

                cv.imshow('Input', input)
                cv.imshow('Target', target)

                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return

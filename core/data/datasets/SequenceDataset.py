import os
import cv2 as cv
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from ..transforms.transforms import (
    ConvertFromInts,
    Clip,
    Normalize,
    ToTensor,
    TransformCompose,
    ConvertColor,
    MakeDivisibleBy,
    RandomResidualsCutPatch
)


class SequenceDataset(Dataset):
    def __init__(self, root_dir, cfg, is_train: bool = True, to_tensor: bool = True):
        self.cfg = cfg
        self.root_dir = root_dir
        self.divisible_by = cfg.INPUT.MAKE_DIVISIBLE_BY
        self.masks_dirname_template = "masks"
        self.resids_dirname_template = cfg.DATASET.SUBDIR_RESIDS
        self.inputs_dirname_template = cfg.DATASET.SUBDIR_INPUTS
        self.targets_dirname_template = cfg.DATASET.SUBDIR_TARGETS
        self.read_masks = cfg.DATASET.READ_MASKS
        self.read_resids = cfg.DATASET.READ_RESIDS
        self.seq_length = cfg.DATASET.SEQUENCE_LENGTH
        self.seq_stride = cfg.DATASET.SEQUENCE_STRIDE
        self.sequences = self.read_sequences(self.root_dir, self.seq_length * self.seq_stride)
        self.transforms = self.build_transforms(self.divisible_by, is_train, to_tensor)

    def __len__(self):
        return len(self.sequences)
    
    def read_sequences(self, root: str, min_length: int):
        seqs = sorted(glob(root + "/*/*"))

        seqs_filtered = []
        for s in seqs:
            inputs_len = len(glob(os.path.join(s, self.inputs_dirname_template, "*")))
            targets_len = len(glob(os.path.join(s, self.targets_dirname_template, "*")))
            masks_len = len(glob(os.path.join(s, self.masks_dirname_template, "*")))
            resids_len = len(glob(os.path.join(s, self.resids_dirname_template, "*")))

            assert inputs_len == targets_len
            if self.read_masks: assert inputs_len == masks_len
            if self.read_resids: assert inputs_len == resids_len

            if inputs_len >= min_length:
                seqs_filtered.append(s)

        return seqs_filtered

    def build_transforms(self, div_by: int = 1, is_train: bool = True, to_tensor: bool = True):
        if is_train:
            transform = [
                MakeDivisibleBy(div_by),
                ConvertColor('BGR', 'RGB'),
                # RandomResidualsCutPatch(0.25, 0.5, 0.1),
                # RandomResidualsCutPatch(1.0, 1.0, 0.33),
                ConvertFromInts(),
                Clip()
            ]
        else:
            transform = [
                MakeDivisibleBy(div_by),
                ConvertColor('BGR', 'RGB'),
                ConvertFromInts(),
                Clip()
            ]

        if to_tensor:
            transform = transform + [Normalize(), ToTensor()]

        transform = TransformCompose(transform)
        return transform

    def __getitem__(self, idx):
        # Scan files
        seq_path = self.sequences[idx]
        input_paths = glob(os.path.join(seq_path, self.inputs_dirname_template, "*"))
        target_paths = glob(os.path.join(seq_path, self.targets_dirname_template, "*"))
        if self.read_masks:
            mask_paths = glob(os.path.join(seq_path, self.masks_dirname_template, "*"))
        if self.read_resids:
            resids_paths = glob(os.path.join(seq_path, self.resids_dirname_template, "*"))

        # Reduce sequence length
        input_paths = input_paths[:self.seq_stride * self.seq_length:self.seq_stride]
        target_paths = target_paths[:self.seq_stride * self.seq_length:self.seq_stride]
        if self.read_masks:
            mask_paths = mask_paths[:self.seq_stride * self.seq_length:self.seq_stride]
        if self.read_resids:
            resids_paths = resids_paths[:self.seq_stride * self.seq_length:self.seq_stride]

        # Read images
        inputs, targets, masks, resids = [], [], [], []
        for i in range(len(input_paths)):
            input = cv.imread(input_paths[i])
            target = cv.imread(target_paths[i])

            if self.read_masks:
                mask = cv.imread(mask_paths[i], cv.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
            else:
                mask = np.full(shape=(*target.shape[0:2], 1), fill_value=255)

            if self.read_resids:
                resid = cv.imread(resids_paths[i])
            else:
                resid = np.full(shape=target.shape, fill_value=127)

            # (1, H, W, C)
            inputs.append(np.expand_dims(input, axis=0))
            targets.append(np.expand_dims(target, axis=0))
            masks.append(np.expand_dims(mask, axis=0))
            resids.append(np.expand_dims(resid, axis=0))

        # (T, H, W, C)
        input_seq = np.concatenate(inputs, axis=0)
        target_seq = np.concatenate(targets, axis=0)
        mask_seq = np.concatenate(masks, axis=0)
        resid_seq = np.concatenate(resids, axis=0)

        # Apply transforms
        if self.transforms:
            input_seq, target_seq, mask_seq, resid_seq = self.transforms(input_seq, target_seq, mask_seq, resid_seq)

        return input_seq, target_seq, mask_seq, resid_seq # (T, C, H, W)

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            input_seq, target_seq, mask_seq, resid_seq = self.__getitem__(i)

            for input, target, mask, resid in zip(input_seq, target_seq, mask_seq, resid_seq):
                input = (input.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                target = (target.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                mask = (mask.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                resid = (resid.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

                cv.imshow('Input', input)
                cv.imshow('Target', target)
                cv.imshow('Mask', mask)
                cv.imshow('Resid', resid)

                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return

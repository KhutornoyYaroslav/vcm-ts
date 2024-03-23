import logging
import time

import torch
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DistributedSampler,
    DataLoader
)

from .datasets import build_dataset


def create_distributed_loader(dataset: Dataset,
                              shuffle: bool,
                              batch_size: int,
                              num_workers: int = 1,
                              pin_memory: bool = True,
                              seed: int = 0):
    if shuffle:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=True)
        data_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, sampler=sampler,
                                 pin_memory=pin_memory, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
        data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=pin_memory)

    return data_loader


def create_loader(dataset: Dataset,
                  shuffle: bool,
                  batch_size: int,
                  num_workers: int = 1,
                  pin_memory: bool = True):
    if shuffle:
        seed = int(time.time())
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=pin_memory)

    return data_loader


def make_data_loader(cfg, seed: int, is_train: bool = True, is_multi_gpu: bool = False) -> DataLoader:
    logger = logging.getLogger('CORE')

    if is_train:
        cfg_dataset_dirs = cfg.DATASET.TRAIN_ROOT_DIRS
        cfg_dataset_lists = cfg.DATASET.TRAIN_SUBDIR_LISTS
    else:
        cfg_dataset_dirs = cfg.DATASET.TEST_ROOT_DIRS
        cfg_dataset_lists = cfg.DATASET.TEST_SUBDIR_LISTS

    # Create datasets
    datasets = []
    for root_dir, lists in zip(cfg_dataset_dirs, cfg_dataset_lists):
        dataset = build_dataset(cfg.DATASET.TYPE, root_dir, cfg, lists, is_train=is_train)
        logger.info("Loaded dataset from '{0}'. Size: {1}".format(root_dir, len(dataset)))
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)

    # Create data loader
    batch_size = len(cfg.SOLVER.LAMBDAS)
    shuffle = is_train
    if is_multi_gpu:
        data_loader = create_distributed_loader(dataset, shuffle, batch_size, cfg.DATA_LOADER.NUM_WORKERS,
                                                cfg.DATA_LOADER.PIN_MEMORY, seed)
    else:
        data_loader = create_loader(dataset, shuffle, batch_size, cfg.DATA_LOADER.NUM_WORKERS,
                                    cfg.DATA_LOADER.PIN_MEMORY)

    return data_loader

from .SequenceDataset import SequenceDataset


_DATASET_TYPES = {
    "SequenceDataset": SequenceDataset
}


def build_dataset(type, root_dir, cfg, is_train):
    dataset = _DATASET_TYPES[type]
    return dataset(root_dir, cfg, is_train)

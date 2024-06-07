from .sequence_dataset import SequenceDataset


_DATASET_TYPES = {
    "SequenceDataset": SequenceDataset
}


def build_dataset(type, root_dir, cfg, dir_list, is_train):
    dataset = _DATASET_TYPES[type]
    return dataset(root_dir, cfg, dir_list, is_train)

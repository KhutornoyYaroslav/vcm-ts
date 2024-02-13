from .SequenceDataset import SequenceDataset
from .DCVCSequenceDataset import DCVCSequenceDataset


_DATASET_TYPES = {
    "SequenceDataset": SequenceDataset,
    "DCVCSequenceDataset": DCVCSequenceDataset,
}


def build_dataset(type, root_dir, cfg, is_train):
    dataset = _DATASET_TYPES[type]
    return dataset(root_dir, cfg, is_train)

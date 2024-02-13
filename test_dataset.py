import argparse
import os

from core.config import dcvc_cfg as cfg
from core.data.datasets import build_dataset


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Testing dataset')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/dcvc_cfg.yaml",
                        help="Path to config file")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Check dataset
    datasets = build_dataset(cfg.DATASET.TYPE, cfg.DATASET.TRAIN_ROOT_DIRS[0], cfg, is_train=True)
    datasets.visualize(1)


if __name__ == '__main__':
    main()

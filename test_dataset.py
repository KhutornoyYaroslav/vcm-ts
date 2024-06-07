import argparse

from core.config import cfg as cfg
from core.data.datasets import build_dataset


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Testing dataset')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Check dataset
    datasets = build_dataset(cfg.DATASET.TYPE, cfg.DATASET.TRAIN_ROOT_DIRS[0], cfg, cfg.DATASET.TRAIN_SUBDIR_LISTS[0],
                             True)
    print('Sequences length = ' + str(len(datasets.sequences)))
    datasets.visualize(25)


if __name__ == '__main__':
    main()

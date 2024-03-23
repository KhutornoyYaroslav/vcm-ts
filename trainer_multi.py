import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.config import cfg as cfg
from core.data import make_data_loader
from core.engine.train_multi import do_train
from core.utils import dist_util
from core.utils.logger import setup_logger


def init_distributed():
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    print(f"Running DDP on rank {rank}.")

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def train_model(cfg, args):
    # Create data loader
    data_loader = make_data_loader(cfg, args.seed, is_train=True, is_multi_gpu=True)

    arguments = {"epoch": 0}

    dist.barrier()
    # Train model
    model = do_train(cfg, data_loader, arguments, args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main(seed):
    # Create argument parser
    parser = argparse.ArgumentParser(description='DCVC Video Compression Model Training With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--save-step', dest="save_step", required=False, type=int, default=1,
                        help='Save checkpoint every save_step')
    parser.add_argument('--eval-step', dest="eval_step", required=False, type=int, default=1,
                        help='Evaluate datasets every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use-tensorboard', dest="use_tensorboard", required=False, default=True, type=str2bool,
                        help='Use tensorboard summary writer')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    NUM_GPUS = int(os.environ['WORLD_SIZE'])
    args.distributed = True
    args.num_gpus = NUM_GPUS
    args.seed = seed

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create output directory
    if dist_util.is_main_process():
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create logger
    logger = setup_logger("CORE", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(NUM_GPUS))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create config backup
    if dist_util.is_main_process():
        with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
            cfg_dump.write(str(cfg))

    # Train model
    model = train_model(cfg, args)


if __name__ == '__main__':
    init_distributed()
    seed = int(time.time())
    main(seed)

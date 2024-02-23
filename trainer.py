import os
import torch
import logging
import argparse
from core.utils import dist_util
from core.config import cfg as cfg
from core.engine.train import do_train
from core.data import make_data_loader
from core.utils.logger import setup_logger
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer
from core.solver import make_optimizer, make_lr_scheduler


def train_model(cfg, args):
    logger = logging.getLogger('CORE')
    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    # Create data loader
    data_loader = make_data_loader(cfg, is_train=True)

    # Create optimizer
    optimizer = make_optimizer(cfg, model, args.num_gpus)
    scheduler = None  # no scheduler, lr changes in train function

    # Create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.is_main_process()
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)

    # Init DMC by default weights
    # extra_checkpoint_data = checkpointer.load('pretrained/acmmm2022_video_psnr.pth')
    extra_checkpoint_data = checkpointer.load('pretrained/spynet.pth')
    arguments.update(extra_checkpoint_data)

    # Train model
    model = do_train(cfg, model, data_loader, optimizer, scheduler, checkpointer, device, arguments, args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Image/Video Codec Model Training With PyTorch')
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
    NUM_GPUS = 1
    args.distributed = False
    args.num_gpus = NUM_GPUS

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create logger
    logger = setup_logger("CORE", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(NUM_GPUS))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create config backup
    with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
        cfg_dump.write(str(cfg))

    # Train model
    model = train_model(cfg, args)


if __name__ == '__main__':
    main()

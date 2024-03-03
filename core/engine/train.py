import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.data import make_data_loader, make_object_detection_data_loader
from core.utils import dist_util
from core.utils.tensorboard import add_best_and_worst_sample, add_metrics
from .validation import eval_dataset


def do_eval(cfg, model, forward_method, loss_dist_key, loss_rate_keys, p_frames, seed, stage, perceptual_loss_key):
    torch.cuda.empty_cache()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, seed, False)
    object_detection_loader = None
    if cfg.DATASET.METADATA_PATH and cfg.DATASET.TEST_OD_ROOT_DIRS:
        object_detection_loader = make_object_detection_data_loader(cfg)
    model.eval()
    result_dict = eval_dataset(model, forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, cfg,
                               object_detection_loader, stage, perceptual_loss_key)

    torch.cuda.empty_cache()
    return result_dict


def calc_max_epoch(cfg):
    for stage_params in cfg.SOLVER.STAGES:
        assert len(stage_params) == 8

    epoch_counter = 0
    for i in range(len(cfg.SOLVER.STAGES)):
        epoch_counter += int(cfg.SOLVER.STAGES[i][6])

    return epoch_counter


def get_stage_params(cfg,
                     model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     epoch: int):
    """
    Evaluates parameters of current training stage.
    List of parameters from configuration file for each stage:
        0 - p_frames
        1 - trainable modules
        2 - forward method
        3 - loss dist
        4 - loss rate
        5 - lr
        6 - epochs

    Parameters:
        cfg : config
            Main configuration parameters.
        model : torch.nn.Module
            Model to train. Need to change trainable modules.
        optimizer : torch.optim.Optimizer
            Optimizer to update model parameters. Need to change learning rate.
        epoch : int
            Current epoch.

    Returns:
        params : dict
            Dict of current training stage parameters.
    """

    result = {
        'stage': None,
        'p_frames': None,
        'forward_method': None,
        'loss_dist_key': None,
        'loss_rate_keys': None,
        'perceptual_loss': None
    }

    # Check number of stage parameters
    for stage_params in cfg.SOLVER.STAGES:
        assert len(stage_params) == 8

    # Get current stage
    epoch_counter = 0
    for i in range(len(cfg.SOLVER.STAGES)):
        epoch_counter += int(cfg.SOLVER.STAGES[i][6])
        if epoch < epoch_counter:
            result['stage'] = i
            break

    stage_params = cfg.SOLVER.STAGES[result['stage']]

    # P-frames number
    result['p_frames'] = int(stage_params[0])
    assert 0 < result['p_frames'] < cfg.DATASET.SEQUENCE_LENGTH, "Invalid 'p_frames' stage parameter"

    # Modules to train
    if stage_params[1] == 'me' and stage_params[4] == 'none':
        model.activate_modules_inter_dist()
    elif stage_params[1] == 'me' and stage_params[4] == 'me':
        model.activate_modules_inter_dist_rate()
    elif stage_params[1] == 'rec' and stage_params[4] == 'none':
        model.activate_modules_recon_dist()
    elif stage_params[1] == 'rec' and stage_params[4] == 'rec':
        model.activate_modules_recon_dist_rate()
    elif stage_params[1] == 'all' and stage_params[4] == 'all':
        model.activate_modules_all()
    else:
        raise SystemError('Invalid pair of part and loss rate')
    # modules for perceptual loss is always eval
    model.dmc.vgg.eval()
    model.dmc.rcnn.eval()

    # Train method
    if stage_params[2] == 'single':
        result['forward_method'] = 'single'
    elif stage_params[2] == 'cascade':
        result['forward_method'] = 'cascade'
    else:
        raise SystemError('Invalid loss type')

    # Loss dist key
    if stage_params[3] == 'me':
        result['loss_dist_key'] = "me_mse"
    elif stage_params[3] == 'rec':
        result['loss_dist_key'] = "mse"
    else:
        raise SystemError('Invalid loss dist')

    # Loss rate keys
    if stage_params[4] == 'none':
        result['loss_rate_keys'] = []
    elif stage_params[4] == 'me':
        result['loss_rate_keys'] = ["bpp_mv_y", "bpp_mv_z"]
    elif stage_params[4] == 'rec':
        result['loss_rate_keys'] = ["bpp_y", "bpp_z"]
    elif stage_params[4] == 'all':
        result['loss_rate_keys'] = ["bpp_mv_y", "bpp_mv_z", "bpp_y", "bpp_z"]
    else:
        raise SystemError('Invalid loss rate')

    # Learning rate
    optimizer.param_groups[0]["lr"] = float(stage_params[5])

    # Perceptual loss
    result['perceptual_loss'] = stage_params[7]

    return result


def do_train(cfg,
             model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             seed,
             arguments,
             args):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # Set model to train mode
    model.train()

    # Create tensorboard writer
    save_to_disk = dist_util.is_main_process()
    if args.use_tensorboard and save_to_disk:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # Prepare to train
    iters_per_epoch = len(data_loader)
    max_epoch = calc_max_epoch(cfg)
    total_steps = iters_per_epoch * max_epoch
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps,
                                                                                       start_epoch))

    # Epoch loop
    for epoch in range(start_epoch, max_epoch):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%9s' * 6 + '%37s' * 2) % ('Epoch', 'stage', 'gpu_mem', 'lr', 'loss', 'perc_loss', 'bpp', 'psnr'))

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Iteration loop
        stats = {
            'loss_sum': 0,
            'perceptual_loss': 0,
            'bpp': 0,
            'mse_sum': 0,
            'psnr': 0,
            'lr': 0.0,
            'stage': 0,
            'best_samples': [],
            'worst_samples': []
        }

        best_samples = [[] for _ in range(len(cfg.SOLVER.LAMBDAS))]
        worst_samples = [[] for _ in range(len(cfg.SOLVER.LAMBDAS))]

        stage_params = get_stage_params(cfg, model, optimizer, epoch)

        total_iterations = 0
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            # Get data
            input, _ = data_entry  # (N, T, C, H, W)
            input = input.cuda()

            # Optimize model
            outputs = model(stage_params['forward_method'],
                            input,
                            stage_params['loss_dist_key'],
                            stage_params['loss_rate_keys'],
                            stage_params['p_frames'],
                            stage_params['perceptual_loss'],
                            optimizer=optimizer)
            total_iterations += outputs['single_forwards']

            # Update stats
            stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (T-1) -> (1)
            stats['perceptual_loss'] += torch.mean(outputs['perceptual_loss'], -1).item()  # (T-1) -> (1)
            stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
            stats['mse_sum'] += 0  # TODO:
            stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            bpp = stats['bpp'] / total_iterations
            bpp = [f'{x:.2f}' for x in bpp]
            psnr = 10 * np.log10(1.0 / (stats['psnr'] / total_iterations))
            psnr = [f'{x:.1f}' for x in psnr]
            s = ('%9s' * 3 + '%9.4g' * 3 + '%37s' * 2) % ('%g/%g' % (epoch + 1, max_epoch),
                                                          ('%g' % (stage_params['stage'] + 1)),
                                                          mem,
                                                          optimizer.param_groups[0]["lr"],
                                                          stats['loss_sum'] / total_iterations,
                                                          stats['perceptual_loss'] / total_iterations,
                                                          ", ".join(bpp),
                                                          ", ".join(psnr)
                                                          )
            pbar.set_description(s)

            add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

        stats['loss_sum'] /= total_iterations
        stats['perceptual_loss'] /= total_iterations
        stats['bpp'] /= total_iterations
        stats['mse_sum'] /= total_iterations
        stats['psnr'] /= total_iterations
        stats['lr'] = optimizer.param_groups[0]["lr"]
        stats['stage'] = stage_params['stage'] + 1
        stats['best_samples'] = best_samples
        stats['worst_samples'] = worst_samples

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Do evaluation
        if (args.eval_step > 0) and (epoch % args.eval_step == 0) and len(cfg.DATASET.TEST_ROOT_DIRS):
            print('\nEvaluation ...')
            result_dict = do_eval(cfg,
                                  model,
                                  stage_params['forward_method'],
                                  stage_params['loss_dist_key'],
                                  stage_params['loss_rate_keys'],
                                  stage_params['p_frames'],
                                  seed,
                                  stage_params['stage'] + 1,
                                  stage_params['perceptual_loss'])

            print(('\n' + 'Evaluation results:' + '%9s' * 2 + '%37s' * 3) % ('loss', 'perc_loss', 'bpp', 'psnr', 'mAP'))
            bpp_print = [f'{x:.2f}' for x in result_dict['bpp']]
            psnr = 10 * np.log10(1.0 / (result_dict['psnr']))
            psnr_print = [f'{x:.1f}' for x in psnr]
            mean_ap_print = [f'{x:.1f}' for x in result_dict['mean_ap']]
            print('                   ' + ('%9.4g' * 2 + '%37s' * 3) %
                  (result_dict['loss_sum'],
                   result_dict['perceptual_loss'],
                   ", ".join(bpp_print),
                   ", ".join(psnr_print),
                   ", ".join(mean_ap_print))
                  )

            add_metrics(cfg, summary_writer, result_dict, global_step, is_train=False)

            model.train()

        # Save epoch results
        if epoch % args.save_step == 0:
            add_metrics(cfg, summary_writer, stats, global_step, is_train=True)

            checkpointer.save("model_{:06d}".format(global_step), **arguments)

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model

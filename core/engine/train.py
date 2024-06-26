import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DCVC_HEM.src.models.image_model import IntraNoAR
from DCVC_HEM.src.utils.common import interpolate_log
from DCVC_HEM.src.utils.stream_helper import get_state_dict
from core.data import make_data_loader, make_object_detection_data_loader
from core.utils import dist_util
from core.utils.tensorboard import add_best_and_worst_sample, add_metrics
from .validation import eval_dataset


def do_eval(cfg, model, forward_method, loss_dist_key, loss_rate_keys, p_frames, seed, stage, perceptual_loss,
            i_frame_net, i_frame_q_scales):
    torch.cuda.empty_cache()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, seed, False)
    object_detection_loader = None
    if cfg.DATASET.METADATA_PATH and cfg.DATASET.TEST_OD_ROOT_DIRS:
        object_detection_loader = make_object_detection_data_loader(cfg)
    model.eval()
    result_dict = eval_dataset(model, forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, cfg,
                               object_detection_loader, stage, perceptual_loss, i_frame_net, i_frame_q_scales)

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
        7 - perceptual loss model

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
    if stage_params[7] == 'true':
        perceptual_loss = True
    elif stage_params[7] == 'false':
        perceptual_loss = False
    else:
        raise SystemError('Invalid perceptual loss usage (true or false)')
    result['perceptual_loss'] = perceptual_loss

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
    model.perceptual_loss.eval()

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

    # Initialize i frame net
    if cfg.MODEL.I_FRAME_PRETRAINED_WEIGHTS != "":
        rate_count = len(cfg.SOLVER.LAMBDAS)
        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(cfg.MODEL.I_FRAME_PRETRAINED_WEIGHTS)
        if len(i_frame_q_scales) == rate_count:
            pass
        else:
            max_q_scale = i_frame_q_scales[0]
            min_q_scale = i_frame_q_scales[-1]
            i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_count)

        i_state_dict = get_state_dict(cfg.MODEL.I_FRAME_PRETRAINED_WEIGHTS)
        i_frame_net = IntraNoAR()
        i_frame_net.load_state_dict(i_state_dict, strict=False)
        i_frame_net = i_frame_net.cuda()
        i_frame_net.eval()
    else:
        i_frame_net = None
        i_frame_q_scales = None

    # Epoch loop
    for epoch in range(start_epoch, max_epoch):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%10s' * 6 + '%37s' * 2) % ('Epoch', 'stage', 'gpu_mem', 'loss', 'dist', 'p_dist',
                                                  'bpp', 'psnr'))

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Iteration loop
        stats = {
            'loss_sum': 0,
            'dist': 0,
            'p_dist': 0,
            'bpp': 0,
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
            global_step = epoch * iters_per_epoch + iteration + 1

            # Get data
            input, target = data_entry  # (N, T, C, H, W)
            input = input.cuda()
            target = target.cuda()

            # Optimize model
            outputs = model(stage_params['forward_method'],
                            input,
                            target,
                            stage_params['loss_dist_key'],
                            stage_params['loss_rate_keys'],
                            stage_params['p_frames'],
                            stage_params['perceptual_loss'],
                            optimizer=optimizer,
                            i_frame_net=i_frame_net,
                            i_frame_q_scales=i_frame_q_scales)
            total_iterations += outputs['single_forwards']

            # Update stats
            stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (N, T-1) -> (1)
            stats['dist'] += torch.mean(torch.sum(outputs['dist'], -1)).item()  # (N, T-1) -> (1)
            stats['p_dist'] += torch.mean(torch.sum(outputs['p_dist'], -1)).item()  # (N, T-1) -> (1)
            stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
            stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            bpp = stats['bpp'] / total_iterations
            bpp = [f'{x:.2f}' for x in bpp]
            psnr = 10 * np.log10(1.0 / (stats['psnr'] / total_iterations))
            psnr = [f'{x:.1f}' for x in psnr]
            s = ('%10s' * 3 + '%10.4g' * 3 + '%37s' * 2) % ('%g/%g' % (epoch + 1, max_epoch),
                                                            ('%g' % (stage_params['stage'] + 1)),
                                                            mem,
                                                            stats['loss_sum'] / total_iterations,
                                                            stats['dist'] / total_iterations,
                                                            stats['p_dist'] / total_iterations,
                                                            ", ".join(bpp),
                                                            ", ".join(psnr)
                                                            )
            pbar.set_description(s)

            add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

        stats['loss_sum'] /= total_iterations
        stats['dist'] /= total_iterations
        stats['p_dist'] /= total_iterations
        stats['bpp'] /= total_iterations
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
                                  stage_params['perceptual_loss'],
                                  i_frame_net,
                                  i_frame_q_scales)

            print(('\n' + 'Evaluation results:' + '%10s' * 3 + '%37s' * 3) % ('loss', 'dist', 'p_dist', 'bpp', 'psnr',
                                                                              'mAP'))
            bpp_print = [f'{x:.2f}' for x in result_dict['bpp']]
            psnr = 10 * np.log10(1.0 / (result_dict['psnr']))
            psnr_print = [f'{x:.1f}' for x in psnr]
            mean_ap_print = [f'{x:.1f}' for x in result_dict['mean_ap']]
            print('                   ' + ('%10.4g' * 3 + '%37s' * 3) %
                  (result_dict['loss_sum'],
                   result_dict['dist'],
                   result_dict['p_dist'],
                   ", ".join(bpp_print),
                   ", ".join(psnr_print),
                   ", ".join(mean_ap_print))
                  )

            add_metrics(cfg, summary_writer, result_dict, global_step, is_train=False)

            model.train()
            model.perceptual_loss.eval()

        # Save epoch results
        if epoch % args.save_step == 0:
            add_metrics(cfg, summary_writer, stats, global_step, is_train=True)

            checkpointer.save("model_{:06d}".format(global_step), **arguments)

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model

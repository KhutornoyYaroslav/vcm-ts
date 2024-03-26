import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from core.utils import dist_util
from .validation import eval_dataset
from torchvision.utils import make_grid
from core.data import make_data_loader
from torch.utils.tensorboard import SummaryWriter
from core.utils.tensorboard import add_best_and_worst_sample, add_metrics
import torch.distributed as dist
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer
from core.solver import make_optimizer
import cv2 as cv


def do_eval(cfg, model, forward_method, loss_dist_key, loss_rate_keys, p_frames, seed):
    torch.cuda.empty_cache()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, seed, False, True)
    model.eval()
    result_dict = eval_dataset(model, forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, cfg)

    torch.cuda.empty_cache()
    return result_dict


def calc_max_epoch(cfg):
    for stage_params in cfg.SOLVER.STAGES:
        assert len(stage_params) == 7

    epoch_counter = 0
    for i in range(len(cfg.SOLVER.STAGES)):
        epoch_counter += int(cfg.SOLVER.STAGES[i][6])

    return epoch_counter


def get_current_stage(cfg, epoch):
    epoch_counter = 0
    for i in range(len(cfg.SOLVER.STAGES)):
        epoch_counter += int(cfg.SOLVER.STAGES[i][6])
        if epoch < epoch_counter:
            return i


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
        'loss_rate_keys': None
    }

    # Check number of stage parameters
    for stage_params in cfg.SOLVER.STAGES:
        assert len(stage_params) == 7

    # Get current stage
    result['stage'] = get_current_stage(cfg, epoch)

    stage_params = cfg.SOLVER.STAGES[result['stage']]

    # P-frames number
    result['p_frames'] = int(stage_params[0])
    assert 0 < result['p_frames'] < cfg.DATASET.SEQUENCE_LENGTH, "Invalid 'p_frames' stage parameter"

    # Modules to train
    if stage_params[1] == 'me' and stage_params[4] == 'none':
        model.module.activate_modules_inter_dist()
    elif stage_params[1] == 'me' and stage_params[4] == 'me':
        model.module.activate_modules_inter_dist_rate()
    elif stage_params[1] == 'rec' and stage_params[4] == 'none':
        model.module.activate_modules_recon_dist()
    elif stage_params[1] == 'rec' and stage_params[4] == 'rec':
        model.module.activate_modules_recon_dist_rate()
    elif stage_params[1] == 'all' and stage_params[4] == 'all':
        model.module.activate_modules_all()
    else:
        raise SystemError('Invalid pair of part and loss rate')

    # Train method
    if stage_params[2] == 'single_multi':
        result['forward_method'] = 'single_multi'
    elif stage_params[2] == 'cascade_multi':
        result['forward_method'] = 'cascade_multi'
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

    return result


def init_model(cfg, logger, arguments):
    # Create model
    model = build_model(cfg).cuda()
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create optimizer
    num_gpus = int(os.environ['WORLD_SIZE'])
    optimizer = make_optimizer(cfg, model, num_gpus)

    # Create checkpointer
    save_to_disk = dist_util.is_main_process()
    checkpointer = CheckPointer(model, optimizer, None, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)
    arguments.update(extra_checkpoint_data)

    return model, optimizer, checkpointer


def reinit_model(model, optimizer, checkpointer, cfg, logger, arguments):
    del checkpointer
    del optimizer
    del model
    torch.cuda.empty_cache()

    return init_model(cfg, logger, arguments)


def single_step(input, model, stage_params, dpb, optimizer, t_i, outputs):
    input_seqs = []
    decod_seqs = []
    input_seqs.append(input[:, t_i])
    decod_seqs.append(input[:, t_i])

    loss_list = []

    # Forward P-frames
    for p_idx in range(0, stage_params['p_frames']):
        optimizer.zero_grad()
        result = model(stage_params['forward_method'],
                       input[:, t_i + 1 + p_idx],
                       stage_params['loss_dist_key'],
                       stage_params['loss_rate_keys'],
                       dpb=dpb)
        loss_to_opt = result['loss_to_opt']
        loss_to_opt.backward()
        optimizer.step()

        dpb = result['dpb']

        loss_list.append(result['loss'])

        # rank = int(os.environ["RANK"])
        # print(f'Final ({rank}) {t_i} {p_idx}: {model.module.dmc.mv_encoder[6].weight.grad[0, 0]}')

        outputs['rate'].append(result['rate'])  # (N)
        outputs['dist'].append(result['dist'])  # (N)
        outputs['loss'].append(result['loss'])  # (N)
        outputs['single_forwards'] += 1
        input_seqs.append(result['input_seqs'])
        decod_seqs.append(result['decod_seqs'])

    loss_seq = torch.stack(loss_list, -1)  # (N, p_frames)
    loss_seq = torch.mean(loss_seq, -1)  # (N, p_frames) -> (N)

    outputs['loss_seq'].append(loss_seq)  # (N)
    outputs['input_seqs'].append(torch.stack(input_seqs, -1))  # (N, p_frames + 1)
    outputs['decod_seqs'].append(torch.stack(decod_seqs, -1))  # (N, p_frames + 1)


def cascade_step(input, model, stage_params, dpb, optimizer, t_i, outputs):
    optimizer.zero_grad()
    result = model(stage_params['forward_method'],
                   input,
                   stage_params['loss_dist_key'],
                   stage_params['loss_rate_keys'],
                   dpb=dpb,
                   p_frames=stage_params['p_frames'],
                   t_i=t_i)
    loss_to_opt = result['loss_to_opt']
    loss_to_opt.backward()
    optimizer.step()

    # rank = int(os.environ["RANK"])
    # print(f'Final ({rank}) {t_i}: {model.module.dmc.mv_encoder[6].weight.grad[0, 0]}')

    outputs['rate'].append(result['rate'])  # (N)
    outputs['dist'].append(result['dist'])  # (N)
    outputs['loss'].append(result['loss'])  # (N)
    outputs['loss_seq'].append(result['loss'])  # (N)
    outputs['single_forwards'] += 1

    outputs['input_seqs'].append(result['input_seqs'])  # (N, p_frames + 1)
    outputs['decod_seqs'].append(result['decod_seqs'])  # (N, p_frames + 1)


def do_train(cfg,
             data_loader,
             arguments,
             args):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    model, optimizer, checkpointer = init_model(cfg, logger, arguments)

    # Set model to train mode
    model.train()

    # Create tensorboard writer
    save_to_disk = dist_util.is_main_process()
    if args.use_tensorboard and save_to_disk:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # Prepare to train
    iters_per_epoch = len(data_loader) * args.num_gpus
    max_epoch = calc_max_epoch(cfg)
    total_steps = iters_per_epoch * max_epoch
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps,
                                                                                       start_epoch))
    current_stage = get_current_stage(cfg, start_epoch)
    rank = int(os.environ["RANK"])

    # Epoch loop
    for epoch in range(start_epoch, max_epoch):
        data_loader.sampler.set_epoch(epoch)
        arguments["epoch"] = epoch + 1

        if current_stage != get_current_stage(cfg, epoch):
            model, optimizer, checkpointer = reinit_model(model, optimizer, checkpointer,
                                                          cfg, logger, arguments)
            current_stage = get_current_stage(cfg, epoch)
            dist.barrier()

        # Create progress bar
        if dist_util.is_main_process():
            print(('\n' + '%12s' * 6 + '%25s' * 2) % ('Epoch', 'stage', 'gpu_mem', 'lr', 'loss', 'rank', 'bpp', 'psnr'))

        # Iteration loop
        stats = {
            'loss_sum': 0,
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
        dist.barrier()
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration * args.num_gpus

            # Get data
            input, _ = data_entry  # (N, T, C, H, W)
            input = input.cuda()

            # if iteration == 0:
            #     rank = int(os.environ["RANK"])
            #     for i in range(input.shape[0]):
            #         input_show = input[i][0]
            #         input_show = (input_show.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            #
            #         input_show = cv.cvtColor(input_show, cv.COLOR_RGB2BGR)
            #
            #         cv.imshow(f'GPU_{rank}_input_{i}', input_show)
            #
            #         if cv.waitKey(0) & 0xFF == ord('q'):
            #             continue

            # Optimize model
            n, t, c, h, w = input.shape
            assert 0 < stage_params['p_frames'] < t

            outputs = {
                'rate': [],  # (N, (T - p_frames) * p_frames or T - p_frames)
                'dist': [],  # (N, (T - p_frames) * p_frames or T - p_frames)
                'loss': [],  # (N, (T - p_frames) * p_frames or T - p_frames)
                'loss_seq': [],  # (N, T - p_frames)
                'input_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
                'decod_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
                'single_forwards': 0
            }

            for t_i in range(0, t - stage_params['p_frames']):
                # Initialize I-frame
                dpb = {
                    "ref_frame": input[:, t_i],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None
                }

                if stage_params['forward_method'] == 'single_multi':
                    single_step(input, model, stage_params, dpb, optimizer, t_i, outputs)
                elif stage_params['forward_method'] == 'cascade_multi':
                    cascade_step(input, model, stage_params, dpb, optimizer, t_i, outputs)

            outputs['rate'] = torch.stack(outputs['rate'], -1)  # (N, (T - p_frames) * p_frames or T - p_frames)
            outputs['dist'] = torch.stack(outputs['dist'], -1)  # (N, (T - p_frames) * p_frames or T - p_frames)
            outputs['loss'] = torch.stack(outputs['loss'], -1)  # (N, (T - p_frames) * p_frames or T - p_frames)
            outputs['loss_seq'] = torch.stack(outputs['loss_seq'], -1)  # (N, T - p_frames)
            outputs['input_seqs'] = torch.stack(outputs['input_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
            outputs['input_seqs'] = outputs['input_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)
            outputs['decod_seqs'] = torch.stack(outputs['decod_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
            outputs['decod_seqs'] = outputs['decod_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)

            total_iterations += outputs['single_forwards']

            # Update stats
            stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (T-1) -> (1)
            stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
            stats['mse_sum'] += 0  # TODO:
            stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            bpp = stats['bpp'] / total_iterations
            bpp = [f'{x:.2f}' for x in bpp]
            psnr = 10 * np.log10(1.0 / (stats['psnr'] / total_iterations))
            psnr = [f'{x:.1f}' for x in psnr]
            s = ('%12s' * 3 + '%12.4g' * 3 + '%25s' * 2) % ('%g/%g' % (epoch + 1, max_epoch),
                                                            ('%g' % (stage_params['stage'] + 1)),
                                                            mem,
                                                            optimizer.param_groups[0]["lr"],
                                                            stats['loss_sum'] / total_iterations,
                                                            rank,
                                                            ", ".join(bpp),
                                                            ", ".join(psnr)
                                                            )
            pbar.set_description(s)

            if dist_util.is_main_process():
                add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

        # Receive metrics from gpus
        iterations = [None for _ in range(args.num_gpus)]
        loss_sum = [None for _ in range(args.num_gpus)]
        bpp = [None for _ in range(args.num_gpus)]
        psnr = [None for _ in range(args.num_gpus)]

        dist.gather_object(
            total_iterations,
            iterations if dist_util.is_main_process() else None,
            dst=0
        )
        dist.gather_object(
            stats['loss_sum'],
            loss_sum if dist_util.is_main_process() else None,
            dst=0
        )
        dist.gather_object(
            stats['bpp'],
            bpp if dist_util.is_main_process() else None,
            dst=0
        )
        dist.gather_object(
            stats['psnr'],
            psnr if dist_util.is_main_process() else None,
            dst=0
        )

        if dist_util.is_main_process():
            total_iterations = sum(iterations)
            stats['loss_sum'] = sum(loss_sum)
            stats['bpp'] = np.sum(bpp, axis=0)
            stats['psnr'] = np.sum(psnr, axis=0)

            stats['loss_sum'] /= total_iterations
            stats['bpp'] /= total_iterations
            stats['mse_sum'] /= total_iterations
            stats['psnr'] /= total_iterations
            stats['lr'] = optimizer.param_groups[0]["lr"]
            stats['stage'] = stage_params['stage'] + 1
            stats['best_samples'] = best_samples
            stats['worst_samples'] = worst_samples

        # Do evaluation
        if ((args.eval_step > 0) and (epoch % args.eval_step == 0) and len(cfg.DATASET.TEST_ROOT_DIRS)
                and dist_util.is_main_process()):
            print('\nEvaluation ...')
            forward_method = stage_params['forward_method'][:-len('_multi')]
            result_dict = do_eval(cfg,
                                  model,
                                  forward_method,
                                  stage_params['loss_dist_key'],
                                  stage_params['loss_rate_keys'],
                                  stage_params['p_frames'],
                                  args.seed)

            print(('\n' + 'Evaluation results:' + '%12s' * 2 + '%25s' * 2) % ('loss', 'mse', 'bpp', 'psnr'))
            bpp_print = [f'{x:.2f}' for x in result_dict['bpp']]
            psnr = 10 * np.log10(1.0 / (result_dict['psnr']))
            psnr_print = [f'{x:.1f}' for x in psnr]
            print('                   ' + ('%12.4g' * 2 + '%25s' * 2) %
                  (result_dict['loss_sum'],
                   result_dict['mse_sum'],
                   ", ".join(bpp_print),
                   ", ".join(psnr_print))
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

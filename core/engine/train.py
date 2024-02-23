import os
import cv2 as cv
import torch
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from core.utils import dist_util
from .inference import eval_dataset
from torchvision.utils import make_grid
from core.data import make_data_loader
from core.engine.losses import CharbonnierLoss, MSELoss, FasterRCNNPerceptualLoss
from torch.utils.tensorboard import SummaryWriter


def do_eval(cfg, model, distributed, **kwargs):
    torch.cuda.empty_cache()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, False)
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    result_dict = eval_dataset(cfg, model, data_loader, device)

    torch.cuda.empty_cache()
    return result_dict


def create_tensorboard_image(seq_idx: int, lrs, hrs, preds, frame_idx: int = 0):  # (N, T, C, H, W)
    # HRs (originals)
    hrs_ = hrs.detach().to('cpu')
    hrs_seq = hrs_[seq_idx]

    # LRs (downsampled -> bicubic x4)
    lrs_ = lrs.detach().to('cpu')
    lrs_seq = lrs_[seq_idx]
    t, c, h, w = hrs_seq.shape
    lrs_seq_up = F.interpolate(lrs_seq, size=(w, h), mode='bicubic')
    lrs_seq_up = torch.clip(lrs_seq_up, 0.0, 1.0)

    # PREDs (predicted)
    preds_ = preds.detach().to('cpu')
    preds_seq = preds_[seq_idx]

    save_img = torch.cat([hrs_seq[frame_idx], lrs_seq_up[frame_idx], preds_seq[frame_idx]], dim=2)
    return save_img


def get_stage_params(cfg,
                     model,
                     optimizer,
                     epoch):
    # Assert stage count
    assert (len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.PARTS)
            and len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.LOSS_TYPE)
            and len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.LOSS_DIST)
            and len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.LOSS_RATE)
            and len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.LR)
            and len(cfg.SOLVER.STAGES) == len(cfg.SOLVER.EPOCHS))

    # Stage
    epoch_counter = 0
    for i in range(len(cfg.SOLVER.EPOCHS)):
        epoch_counter += cfg.SOLVER.EPOCHS[i]
        if epoch >= epoch_counter:
            continue
        stage = i
        break

    # P-frames number
    if cfg.SOLVER.STAGES[stage] == 'single':
        p_frames = 1
    elif cfg.SOLVER.STAGES[stage] == 'dual':
        p_frames = 2
    elif cfg.SOLVER.STAGES[stage] == 'multi':
        p_frames = 4
    else:
        raise SystemError('Invalid stage')

    # Modules to train
    if cfg.SOLVER.PARTS[stage] == 'inter' and cfg.SOLVER.LOSS_RATE[stage] == 'none':
        model.activate_modules_inter_dist()
    elif cfg.SOLVER.PARTS[stage] == 'inter' and cfg.SOLVER.LOSS_RATE[stage] == 'me':
        model.activate_modules_inter_dist_rate()
    elif cfg.SOLVER.PARTS[stage] == 'recon' and cfg.SOLVER.LOSS_RATE[stage] == 'none':
        model.activate_modules_recon_dist()
    elif cfg.SOLVER.PARTS[stage] == 'recon' and cfg.SOLVER.LOSS_RATE[stage] == 'rec':
        model.activate_modules_recon_dist_rate()
    elif cfg.SOLVER.PARTS[stage] == 'all' and cfg.SOLVER.LOSS_RATE[stage] == 'all':
        model.activate_modules_all()
    else:
        raise SystemError('Invalid pair of part and loss rate')

    # Train method
    if cfg.SOLVER.LOSS_TYPE[stage] == 'single':
        train_method = model.train_single
    elif cfg.SOLVER.LOSS_TYPE[stage] == 'cascade':
        train_method = model.train_cascade
    else:
        raise SystemError('Invalid loss type')

    # Loss dist key
    if cfg.SOLVER.LOSS_DIST[stage] == 'me':
        loss_dist_key = "me_mse"
    elif cfg.SOLVER.LOSS_DIST[stage] == 'rec':
        loss_dist_key = "mse"
    else:
        raise SystemError('Invalid loss dist')

    # Loss rate keys
    if cfg.SOLVER.LOSS_RATE[stage] == 'none':
        loss_rate_keys = []
    elif cfg.SOLVER.LOSS_RATE[stage] == 'me':
        loss_rate_keys = ["bpp_mv_y", "bpp_mv_z"]
    elif cfg.SOLVER.LOSS_RATE[stage] == 'rec':
        loss_rate_keys = ["bpp_y", "bpp_z"]
    elif cfg.SOLVER.LOSS_RATE[stage] == 'all':
        loss_rate_keys = ["bpp_mv_y", "bpp_mv_z", "bpp_y", "bpp_z"]
    else:
        raise SystemError('Invalid loss rate')

    # Learning rate
    optimizer.param_groups[0]["lr"] = cfg.SOLVER.LR[stage]

    return stage, train_method, p_frames, loss_dist_key, loss_rate_keys


def do_train(cfg,
             model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
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
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps,
                                                                                       start_epoch))

    # Epoch loop
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%12s' * 6 + '%25s' * 2) % ('Epoch', 'stage', 'gpu_mem', 'lr', 'loss', 'mse', 'bpp', 'psnr'))

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Prepare data for tensorboard
        # best_samples, worst_samples = [], []

        # Iteration loop
        stats = {
            'loss_sum': 0,
            'bpp': 0,
            'mse_sum': 0,
            'psnr': 0
        }

        stage, train_method, p_frames, loss_dist_key, loss_rate_keys = get_stage_params(cfg, model, optimizer, epoch)

        total_iterations = 0
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            input, _ = data_entry  # (N, T, C, H, W)

            # Forward data to GPU
            input = input.to(device)

            # Do prediction
            outputs = train_method(input, optimizer, loss_dist_key, loss_rate_keys, p_frames=p_frames)
            total_iterations += outputs['single_forwards']

            # Update stats
            stats['loss_sum'] += torch.sum(outputs['loss']).item()  # (T-1) -> (1)
            stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
            stats['mse_sum'] += 0  # TODO:
            stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            bpp = stats['bpp'] / total_iterations
            bpp = [f'{x:.2f}' for x in bpp]
            psnr = 10 * np.log10(1.0 / (stats['psnr'] / total_iterations))
            psnr = [f'{x:.1f}' for x in psnr]
            s = ('%12s' * 3 + '%12.4g' * 3 + '%25s' * 2) % ('%g/%g' % (epoch, cfg.SOLVER.MAX_EPOCH - 1),
                                                            ('%g' % stage),
                                                            mem,
                                                            optimizer.param_groups[0]["lr"],
                                                            stats['loss_sum'] / total_iterations,
                                                            stats['mse_sum'] / total_iterations,
                                                            ", ".join(bpp),
                                                            ", ".join(psnr)
                                                            )
            pbar.set_description(s)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # # Do evaluation
        # if (args.eval_step > 0) and (epoch % args.eval_step == 0) and len(cfg.DATASET.TEST_ROOT_DIRS):
        #     print('\nEvaluation ...')
        #     result_dict = do_eval(cfg, model, distributed=args.distributed)

        #     print(('\n' + 'Evaluation results:' + '%12s' * 4) % ('loss', 'charb_loss', 'perc_loss', 'psnr'))
        #     psnr = 10 * np.log10(1.0 / (result_dict['mse'] + 1e-9))
        #     print('                   ' + ('%12.4g' * 4) % (result_dict['loss'], result_dict['charbonnier'], result_dict['perception'], psnr))

        #     if summary_writer:
        #         summary_writer.add_scalar('val_losses/loss', result_dict['loss'], global_step=global_step)
        #         summary_writer.add_scalar('val_losses/charb_loss', result_dict['charbonnier'], global_step=global_step)
        #         summary_writer.add_scalar('val_losses/perc_loss', result_dict['perception'], global_step=global_step)
        #         summary_writer.add_scalar('val_losses/psnr', psnr, global_step=global_step)
        #         summary_writer.flush()

        #     model.train()

        # Save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)

            # if summary_writer:
            #     with torch.no_grad():
            #         # Best samples
            #         if len(best_samples):
            #             tb_images = [sample[1] for sample in best_samples]
            #             image_grid = torch.stack(tb_images, dim=0)
            #             image_grid = make_grid(image_grid, nrow=1)
            #             summary_writer.add_image('images/train_best_samples', image_grid, global_step=global_step)

            #         # Worst samples
            #         if len(worst_samples):
            #             tb_images = [sample[1] for sample in worst_samples]
            #             image_grid = torch.stack(tb_images, dim=0)
            #             image_grid = make_grid(image_grid, nrow=1)
            #             summary_writer.add_image('images/train_worst_samples', image_grid, global_step=global_step)

            #         summary_writer.add_scalar('losses/loss', loss_sum / (iteration + 1), global_step=global_step)
            #         summary_writer.add_scalar('losses/charb_loss', charb_loss_sum / (iteration + 1), global_step=global_step)
            #         summary_writer.add_scalar('losses/percept_loss', percept_loss_sum / (iteration + 1), global_step=global_step)
            #         # summary_writer.add_scalar('losses/psnr', 10 * np.log10(1.0 / (mse_loss_sum / (cfg.SOLVER.BATCH_SIZE*iteration + 1))), global_step=global_step)
            #         summary_writer.add_scalar('losses/psnr', 10 * np.log10(1.0 / (mse_loss_sum / (iteration + 1))), global_step=global_step)
            #         summary_writer.add_scalar('lr', optimizers['net'].param_groups[0]['lr'], global_step=global_step)
            #         summary_writer.add_scalar('lr_spynet', optimizers['net'].param_groups[1]['lr'], global_step=global_step)
            #         summary_writer.flush()

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model

# TODO:
# # ################### Best images
# if cfg.TENSORBOARD.BEST_SAMPLES_NUM > 0:
#     with torch.no_grad():
#         losses_ = charb_losses.detach().clone().to('cpu')

#         # Find best metric
#         losses_per_seq = torch.mean(losses_, dim=(1, 2, 3, 4))
#         min_idx = torch.argmin(losses_per_seq).item()
#         best_loss = losses_per_seq[min_idx].item()

#         # Check if better than existing
#         need_save, id_to_remove = True, -1
#         if len(best_samples) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
#             id_to_remove = max(range(len(best_samples)), key=lambda x: best_samples[x][0])
#             if best_loss > best_samples[id_to_remove][0]:
#                 need_save = False

#         # Prepare tensorboard image
#         if need_save:
#             save_img = create_tensorboard_image(min_idx, inputs, targets, outputs)
#             if id_to_remove != -1:
#                 del best_samples[id_to_remove]
#             best_samples.append((best_loss, save_img))
# # ###############################

# # ################### Worst images
# if cfg.TENSORBOARD.WORST_SAMPLES_NUM > 0:
#     with torch.no_grad():
#         losses_ = charb_losses.detach().clone().to('cpu')

#         # Find worst metric
#         losses_per_seq = torch.mean(losses_, dim=(1, 2, 3, 4))
#         max_idx = torch.argmax(losses_per_seq).item()
#         worst_loss = losses_per_seq[max_idx].item()

#         # Check if worse than existing
#         need_save, id_to_remove = True, -1
#         if len(worst_samples) >= cfg.TENSORBOARD.WORST_SAMPLES_NUM:
#             id_to_remove = min(range(len(worst_samples)), key=lambda x: worst_samples[x][0])
#             if worst_loss <= worst_samples[id_to_remove][0]:
#                 need_save = False

#         # Prepare tensorboard image
#         if need_save:
#             save_img = create_tensorboard_image(min_idx, inputs, targets, outputs)
#             if id_to_remove != -1:
#                 del worst_samples[id_to_remove]
#             worst_samples.append((worst_loss, save_img))
# # ###############################

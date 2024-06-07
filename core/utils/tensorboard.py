import numpy as np
import torch
from torchvision.utils import make_grid


def create_tensorboard_image(batch_index, min_index, inputs, outputs):  # (N, T - p_frames, p_frames + 1, C, H, W)
    # Inputs (originals)
    inputs_ = inputs.to('cpu')
    inputs_seq = inputs_[batch_index][min_index]
    common_inputs = inputs_seq[0]
    for i in range(inputs_seq.shape[0] - 1):
        common_inputs = torch.cat([common_inputs, inputs_seq[i + 1]], dim=-1)

    # Outputs (decoded)
    outputs_ = outputs.to('cpu')
    outputs_seq = outputs_[batch_index][min_index]
    common_outputs = outputs_seq[0]
    for i in range(outputs_seq.shape[0] - 1):
        common_outputs = torch.cat([common_outputs, outputs_seq[i + 1]], dim=-1)

    save_img = torch.cat([common_inputs, common_outputs], dim=-2)
    return save_img


def add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples):
    # Best images
    if cfg.TENSORBOARD.BEST_SAMPLES_NUM > 0:
        with torch.no_grad():
            losses = outputs['loss_seq'].clone().to('cpu')
            best_losses = []

            # Find best metric
            min_indexes = torch.argmin(losses, dim=-1).numpy()
            for i in range(losses.shape[0]):
                best_losses.append(losses[i][min_indexes[i]].item())

            # Check if better than existing
            for i in range(losses.shape[0]):
                if len(best_samples[i]) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                    id_to_remove = max(range(len(best_samples[i])), key=lambda x: best_samples[i][x][0])
                    if best_losses[i] < best_samples[i][id_to_remove][0]:
                        # Prepare tensorboard image
                        save_img = create_tensorboard_image(i, min_indexes[i], outputs['input_seqs'],
                                                            outputs['decod_seqs'])
                        # save_img = None
                        if id_to_remove != -1:
                            del best_samples[i][id_to_remove]
                        best_samples[i].append((best_losses[i], save_img))
                else:
                    save_img = create_tensorboard_image(i, min_indexes[i], outputs['input_seqs'],
                                                        outputs['decod_seqs'])
                    best_samples[i].append((best_losses[i], save_img))

    # Worst images
    if cfg.TENSORBOARD.WORST_SAMPLES_NUM > 0:
        with torch.no_grad():
            losses = outputs['loss_seq'].clone().to('cpu')
            worst_losses = []

            # Find best metric
            max_indexes = torch.argmax(losses, dim=-1).numpy()
            for i in range(losses.shape[0]):
                worst_losses.append(losses[i][max_indexes[i]].item())

            # Check if worse than existing
            for i in range(losses.shape[0]):
                if len(worst_samples[i]) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                    id_to_remove = max(range(len(worst_samples[i])), key=lambda x: worst_samples[i][x][0])
                    if worst_losses[i] > worst_samples[i][id_to_remove][0]:
                        # Prepare tensorboard image
                        save_img = create_tensorboard_image(i, max_indexes[i], outputs['input_seqs'],
                                                            outputs['decod_seqs'])
                        # save_img = None
                        if id_to_remove != -1:
                            del worst_samples[i][id_to_remove]
                        worst_samples[i].append((worst_losses[i], save_img))
                else:
                    save_img = create_tensorboard_image(i, max_indexes[i], outputs['input_seqs'],
                                                        outputs['decod_seqs'])
                    worst_samples[i].append((worst_losses[i], save_img))


def add_metrics(cfg, summary_writer, result_dict, global_step, is_train: bool = False):
    if summary_writer is None:
        return

    prefix = 'train' if is_train else 'eval'
    bpp_dict = {}
    psnr_dict = {}
    psnr = 10 * np.log10(1.0 / (result_dict['psnr']))
    for i, l in enumerate(cfg.SOLVER.LAMBDAS):
        bpp_dict[f"lambda_{i + 1}_{l}"] = result_dict['bpp'][i]
        psnr_dict[f"lambda_{i + 1}_{l}"] = psnr[i]
    summary_writer.add_scalar(f'{prefix}_losses/loss', result_dict['loss_sum'], global_step=global_step)
    summary_writer.add_scalar(f'{prefix}_losses/dist', result_dict['dist'], global_step=global_step)
    summary_writer.add_scalar(f'{prefix}_losses/p_dist', result_dict['p_dist'], global_step=global_step)
    summary_writer.add_scalars(f'{prefix}_losses/bpp', bpp_dict, global_step=global_step)
    summary_writer.add_scalars(f'{prefix}_losses/psnr', psnr_dict, global_step=global_step)
    if is_train:
        summary_writer.add_scalar(f'{prefix}_losses/lr', result_dict['lr'], global_step=global_step)
        summary_writer.add_scalar(f'{prefix}_losses/stage', result_dict['stage'], global_step=global_step)
    else:
        mean_ap_dict = {}
        for i, l in enumerate(cfg.SOLVER.LAMBDAS):
            mean_ap_dict[f"lambda_{i + 1}_{l}"] = result_dict['mean_ap'][i]
        summary_writer.add_scalars(f'{prefix}_losses/mean_ap', mean_ap_dict, global_step=global_step)

    with torch.no_grad():
        # Best samples
        if len(result_dict['best_samples'][0]):
            for i, l in enumerate(cfg.SOLVER.LAMBDAS):
                tb_images = [sample[1] for sample in result_dict['best_samples'][i]]
                image_grid = torch.stack(tb_images, dim=0)
                image_grid = make_grid(image_grid, nrow=1)
                summary_writer.add_image(f'images/{prefix}_best_samples_lambda_{i + 1}_{l}', image_grid,
                                         global_step=global_step)

        # Worst samples
        if len(result_dict['worst_samples'][0]):
            for i, l in enumerate(cfg.SOLVER.LAMBDAS):
                tb_images = [sample[1] for sample in result_dict['worst_samples'][i]]
                image_grid = torch.stack(tb_images, dim=0)
                image_grid = make_grid(image_grid, nrow=1)
                summary_writer.add_image(f'images/{prefix}_worst_samples_lambda_{i + 1}_{l}', image_grid,
                                         global_step=global_step)

    summary_writer.flush()

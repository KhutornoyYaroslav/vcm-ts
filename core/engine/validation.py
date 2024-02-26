import logging

import torch
from tqdm import tqdm


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


def eval_dataset(forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, device, cfg):
    logger = logging.getLogger("CORE.inference")

    # Iteration loop
    stats = {
        'loss_sum': 0,
        'bpp': 0,
        'mse_sum': 0,
        'psnr': 0,
        'best_samples': [],
        'worst_samples': []
    }

    sample_count = 0
    best_samples = []
    worst_samples = []
    for data_entry in tqdm(data_loader):
        input, _ = data_entry  # (N, T, C, H, W)

        # Forward images
        with torch.no_grad():
            # Forward data to GPU
            input = input.to(device)

            # Do prediction
            outputs = forward_method(input, None, loss_dist_key, loss_rate_keys, p_frames=p_frames, is_train=False)

        stats['loss_sum'] += torch.sum(torch.mean(outputs['loss'], -1)).item()  # (T-1) -> (1)
        stats['bpp'] += torch.sum(outputs['rate'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        stats['mse_sum'] += 0  # TODO:
        stats['psnr'] += torch.sum(outputs['dist'], -1).cpu().detach().numpy()  # (N, T-1) -> (N)
        sample_count += outputs['single_forwards']

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
                    if len(best_samples) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                        id_to_remove = max(range(len(best_samples)), key=lambda x: best_samples[x][0])
                        if best_losses[i] < best_samples[id_to_remove][0]:
                            # Prepare tensorboard image
                            save_img = create_tensorboard_image(i, min_indexes[i], outputs['input_seqs'],
                                                                outputs['decod_seqs'])
                            # save_img = None
                            if id_to_remove != -1:
                                del best_samples[id_to_remove]
                            best_samples.append((best_losses[i], save_img))
                    else:
                        save_img = create_tensorboard_image(i, min_indexes[i], outputs['input_seqs'],
                                                            outputs['decod_seqs'])
                        best_samples.append((best_losses[i], save_img))

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
                    if len(worst_samples) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                        id_to_remove = max(range(len(worst_samples)), key=lambda x: worst_samples[x][0])
                        if worst_losses[i] > worst_samples[id_to_remove][0]:
                            # Prepare tensorboard image
                            save_img = create_tensorboard_image(i, max_indexes[i], outputs['input_seqs'],
                                                                outputs['decod_seqs'])
                            # save_img = None
                            if id_to_remove != -1:
                                del worst_samples[id_to_remove]
                            worst_samples.append((worst_losses[i], save_img))
                    else:
                        save_img = create_tensorboard_image(i, max_indexes[i], outputs['input_seqs'],
                                                            outputs['decod_seqs'])
                        worst_samples.append((worst_losses[i], save_img))

    # Return results
    stats['loss_sum'] /= sample_count
    stats['bpp'] /= sample_count
    stats['mse_sum'] /= sample_count
    stats['psnr'] /= sample_count
    stats['best_samples'] = best_samples
    stats['worst_samples'] = worst_samples

    return stats

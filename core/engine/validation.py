import logging

import torch
from tqdm import tqdm
from core.utils.tensorboard import add_best_and_worst_sample

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
    best_samples = [[] for _ in range(len(cfg.SOLVER.LAMBDAS))]
    worst_samples = [[] for _ in range(len(cfg.SOLVER.LAMBDAS))]
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

        add_best_and_worst_sample(cfg, outputs, best_samples, worst_samples)

    # Return results
    stats['loss_sum'] /= sample_count
    stats['bpp'] /= sample_count
    stats['mse_sum'] /= sample_count
    stats['psnr'] /= sample_count
    stats['best_samples'] = best_samples
    stats['worst_samples'] = worst_samples

    return stats

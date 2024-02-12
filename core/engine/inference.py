import torch
import logging
from tqdm import tqdm
from core.engine.losses import MSELoss, CharbonnierLoss, FasterRCNNPerceptualLoss


def eval_dataset(cfg, model, data_loader, device):
    logger = logging.getLogger("CORE.inference")

    # Create metrics
    mse_metric = MSELoss(reduction='none')
    charbonnier_metric = CharbonnierLoss(eps=1e-8)
    perception_metric = FasterRCNNPerceptualLoss(device)

    # Iteration loop
    stats = {
        'sample_count': 0.0,
        'loss': 0.0,
        'mse': 0.0,
        'perception': 0.0,
        'charbonnier': 0.0,
    }

    for data_entry in tqdm(data_loader):
        inputs, targets, masks, resids = data_entry # (N, T, C, H, W)

        # Forward images
        with torch.no_grad():
            # Forward data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            # masks = masks.to(device)
            # resids = resids.to(device)

            # Do prediction
            outputs = model(inputs)
            outputs = torch.clip(outputs, 0.0, 1.0)

            # Calculate loss
            mse_losses = mse_metric.forward(outputs, targets)
            charb_losses = charbonnier_metric.forward(outputs, targets)
            percept_losses = perception_metric.forward(outputs.squeeze(0), targets.squeeze(0))

            # Reduce loss
            mse_loss = torch.mean(mse_losses)
            charb_loss = torch.mean(charb_losses)
            percept_loss = torch.mean(percept_losses)

            # Calculate final loss
            loss = charb_loss + cfg.SOLVER.PERCEPTION_LOSS_WEIGHT * percept_loss

        stats['loss'] += loss.item()
        stats['mse'] += mse_loss.item()
        stats['charbonnier'] += charb_loss.item()
        stats['perception'] += percept_loss.item()
        stats['sample_count'] += 1

    # Return results
    stats['loss'] /= stats['sample_count']
    stats['mse'] /= stats['sample_count']
    stats['charbonnier'] /= stats['sample_count']
    stats['perception'] /= stats['sample_count']

    result_dict = {
        'loss': stats['loss'],
        'mse': stats['mse'],
        'charbonnier': stats['charbonnier'],
        'perception': stats['perception'],
    }

    return result_dict

import torch


def make_optimizer(cfg, model: torch.nn.Module, num_gpus: int = 1) -> torch.optim.Optimizer:
    lr_scale = num_gpus
    lr = float(cfg.SOLVER.LR) * lr_scale

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    optimizer = torch.optim.AdamW(params=params, lr=lr, betas=(0.9, 0.99))

    return optimizer

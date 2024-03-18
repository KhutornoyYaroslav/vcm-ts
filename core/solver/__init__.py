import torch


def make_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    lr = float(cfg.SOLVER.LR)

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    optimizer = torch.optim.AdamW(params=params, lr=lr, betas=(0.9, 0.99))

    return optimizer

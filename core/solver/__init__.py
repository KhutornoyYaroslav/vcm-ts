import torch
from typing import Dict
from torch.optim.lr_scheduler import CosineAnnealingLR


def split_parameters(model: torch.nn.Module):
    parameters = {
        # 'aux': {
        #     name
        #     for name, param in model.named_parameters()
        #     if param.requires_grad and name.endswith(".quantiles")
        # },
        # 'spynet': {
        #     name
        #     for name, param in model.named_parameters()
        #     if (param.requires_grad) and (not name.endswith(".quantiles")) and ("spynet" in name)
        # },
        # 'base': {
        #     name
        #     for name, param in model.named_parameters()
        #     if (param.requires_grad) and (not name.endswith(".quantiles")) and ("spynet" not in name)
        # },
        'spynet': {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and ("spynet" in name)
        },
        'base': {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and ("spynet" not in name)
        },
    }

    # Check split correctness
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1

    # inter_params = parameters['base'] & parameters['aux'] & parameters['spynet']
    # union_params = parameters['base'] | parameters['aux'] | parameters['spynet']
    inter_params = parameters['base'] & parameters['spynet']
    union_params = parameters['base'] | parameters['spynet']
    assert len(inter_params) == 0
    assert len(union_params) - total_params == 0

    # Filter parameters by names
    # parameters['aux'] = list(filter(lambda kv: kv[0] in parameters['aux'], model.named_parameters()))
    parameters['base'] = list(filter(lambda kv: kv[0] in parameters['base'], model.named_parameters()))
    parameters['spynet'] = list(filter(lambda kv: kv[0] in parameters['spynet'], model.named_parameters()))

    # Return parameters values
    # parameters['aux'] = [p[1] for p in parameters['aux']]
    parameters['base'] = [p[1] for p in parameters['base']]
    parameters['spynet'] = [p[1] for p in parameters['spynet']]

    return parameters


def make_optimizers(cfg, model: torch.nn.Module, num_gpus: int = 1) -> Dict[str, torch.optim.Optimizer]:
    optimizers = {
        'net': None,
        'aux': None
    }

    # Scale by number of GPUs
    lr_scale = num_gpus if (cfg.MODEL.DEVICE == 'cuda') else 1
    lr = cfg.SOLVER.LR * lr_scale
    # lr_aux = cfg.SOLVER.LR_AUX * lr_scale
    lr_spynet = cfg.SOLVER.LR_SPYNET * lr_scale

    # Split parameters
    param_groups = split_parameters(model)
    assert len(param_groups['base'])

    # NET optimizer
    optim_args = [{'params': param_groups['base'], 'lr': lr}]
    if 'spynet' in param_groups and len(param_groups['spynet']):
        optim_args.append({'params': param_groups['spynet'], 'lr': lr_spynet})
    optimizers['net'] = torch.optim.Adam(params=optim_args, betas=(0.9, 0.99))

    # # AUX optimizer
    # if 'aux' in param_groups and len(param_groups['aux']):
    #     optimizers['aux'] = torch.optim.Adam(param_groups['aux'], lr_aux)

    return optimizers


def make_lr_scheduler(cfg, optimizer): # TODO: optimizerS ?
    # lambda_ = lambda epoch: cfg.SOLVER.LR_LAMBDA ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.MAX_EPOCH, eta_min=1e-7)

    return scheduler

import torch


def quantize_tensor(x: torch.Tensor, bits: int, clip_std: float = None, mask: torch.Tensor = None): # BCHW
    # Calculate tensor mean and std
    x_masked = x if (mask is None) else x.masked_select(mask.to(dtype=torch.bool))

    if clip_std is None:
        x_min, x_max = torch.min(x_masked), torch.max(x_masked)
    else:
        std, mean = torch.std_mean(x_masked)
        std, mean = std.item(), mean.item()
        x_min, x_max = mean - clip_std * std, mean + clip_std * std      

    # Reject outliers
    x = torch.clamp(x, x_min, x_max)

    # Discretize values
    x = torch.round((2 ** bits - 1) * (x - x_min) / (x_max - x_min))

    return x, x_min, x_max


def dequantize_tensor(x: torch.Tensor, x_min: float, x_max: float, bits: int):
    return x * (x_max - x_min) / (2 ** bits - 1) + x_min


def quantize_tensor_per_channel(x: torch.Tensor, bits: int, clip_std: float = None, mask: torch.Tensor = None): # BCHW
    x_ress, x_mins, x_maxs = [], [], []

    for ch_id in range(x.shape[1]):
        x_ch = torch.unsqueeze(x[:, ch_id, :, :], dim=1)
        mask_ch = None if (mask is None) else torch.unsqueeze(mask[:, ch_id, :, :], dim=1)

        x_res, x_min, x_max = quantize_tensor(x_ch, bits, clip_std, mask_ch)
        x_res = x_res.squeeze(dim=1)

        x_ress.append(x_res)
        x_mins.append(x_min)
        x_maxs.append(x_max)

    return torch.stack(x_ress, dim=1), x_mins, x_maxs


def dequantize_tensor_per_channel(x: torch.Tensor, x_mins: list, x_maxs: list, bits: int): # BCHW
    x_res = []
    for idx in range(x.shape[1]):
        x_res.append(dequantize_tensor(x[:, idx, :, :], x_mins[idx], x_maxs[idx], bits))

    return torch.stack(x_res, dim=1)

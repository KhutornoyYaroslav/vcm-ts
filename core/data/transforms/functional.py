import torch
import numpy as np
from torch import Tensor, nn


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def upsample_as_torch(frame: np.array,
                      upsample_module = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)):

    input = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (C, H, W)
    output = upsample_module.forward(input).squeeze(0)
    result = 255 * output.cpu().numpy().astype(np.float32).transpose((1, 2, 0)) # (C, H, W) -> (H, W, C)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def make_array_divisible_by(image: np.array, div_factor: int):
    if (
        not image.ndim in [3, 4] # (HxWxC) or (NxHxWxC)
    ):
        raise ValueError("Expected a 3D or 4D array as input")

    height, width = image.shape[-3: -1]
    rows = height // div_factor + (1 if height % div_factor else 0)
    cols = width // div_factor + (1 if width % div_factor else 0)

    padding = [(0, rows * div_factor - height), (0, cols * div_factor - width), (0, 0)]
    if image.ndim == 4:
        padding.insert(0, (0, 0))
    image = np.pad(image, padding, mode='constant', constant_values=0)
    
    return image


def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in [3, 4, 5]
        # or not tensor.size(-3) == 3
    ):
        raise ValueError(
            # "Expected a 3D, 4D or 5D tensor with shape (NxTxCxHxW), (NxCxHxW) or (CxHxW) as input"
            "Expected a 3D, 4D or 5D tensor as input"
        )


def rgb2ycbcr(rgb: Tensor) -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D, 4D or 5D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D, 4D or 5D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


# def yuv_444_to_420(
#     yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
#     mode: str = "avg_pool",
# ) -> Tuple[Tensor, Tensor, Tensor]:
#     """Convert a 444 tensor to a 420 representation.

#     Args:
#         yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
#             input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
#             of 3 (Nx1xHxW) tensors.
#         mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
#             ``'avg_pool'``

#     Returns:
#         (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
#     """
#     if mode not in ("avg_pool",):
#         raise ValueError(f'Invalid downsampling mode "{mode}".')

#     if mode == "avg_pool":

#         def _downsample(tensor):
#             return F.avg_pool2d(tensor, kernel_size=2, stride=2)

#     if isinstance(yuv, torch.Tensor):
#         y, u, v = yuv.chunk(3, 1)
#     else:
#         y, u, v = yuv

#     return (y, _downsample(u), _downsample(v))


# def yuv_420_to_444(
#     yuv: Tuple[Tensor, Tensor, Tensor],
#     mode: str = "bilinear",
#     return_tuple: bool = False,
# ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
#     """Convert a 420 input to a 444 representation.

#     Args:
#         yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
#             (Nx1xHxW) format
#         mode (str): algorithm used for upsampling: ``'bilinear'`` |
#             | ``'bilinear'`` | ``'nearest'`` Default ``'bilinear'``
#         return_tuple (bool): return input as tuple of tensors instead of a
#             concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
#             tensor (default: False)

#     Returns:
#         (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
#             444
#     """
#     if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
#         raise ValueError("Expected a tuple of 3 torch tensors")

#     if mode not in ("bilinear", "bicubic", "nearest"):
#         raise ValueError(f'Invalid upsampling mode "{mode}".')

#     kwargs = {}
#     if mode != "nearest":
#         kwargs = {"align_corners": False}

#     def _upsample(tensor):
#         return F.interpolate(tensor, scale_factor=2, mode=mode, **kwargs)

#     y, u, v = yuv
#     u, v = _upsample(u), _upsample(v)
#     if return_tuple:
#         return y, u, v
#     return torch.cat((y, u, v), dim=1)

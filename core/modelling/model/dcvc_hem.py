import torch
from torch import nn
from DCVC_HEM.src.models.video_model import DMC


class DCVC_HEM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dmc = DMC(anchor_num=len(cfg.SOLVER.LAMBDAS))

    def forward(self, input: torch.Tensor):
        """
        input: tensor with shape (N, T, C, H, W)
        target: tensor with shape (N, T, C, H, W)
        """
        n, t, c, h, w = input.shape

        # Initialize decoded picture buffer
        dpb = {
            "ref_frame": input[:, 0],
            "ref_feature": None,
            "ref_y": None,
            "ref_mv_y": None
        }

        # Prepare result dict
        result = {
            "bpp_mv_y": [],
            "bpp_mv_z": [],
            "bpp_y": [],
            "bpp_z": [],
            "bpp": [],
            "me_mse": [],
            "mse": [],
            # "ssim": [],
            "bit": [],
            "bit_y": [],
            "bit_z": [],
            "bit_mv_y": [],
            "bit_mv_z": []
        }

        # Process group of pictures
        for t_idx in range(1, t):
            output = self.dmc.forward_one_frame(input[:, t_idx],
                                                dpb,
                                                self.dmc.mv_y_q_scale,
                                                self.dmc.y_q_scale)
            dpb = output['dpb']

            for key, val in output.items():
                if key == 'dpb': continue
                if key in result.keys():
                    result[key].append(val)

        for key, val in result.items():
            result[key] = torch.stack(result[key], dim=-1)

        return result

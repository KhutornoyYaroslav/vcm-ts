import torch
from torch import nn
from typing import List
from DCVC_HEM.src.models.video_model import DMC


class DCVC_HEM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dmc = DMC(anchor_num=len(cfg.SOLVER.LAMBDAS))
        self.lambdas = torch.FloatTensor(cfg.SOLVER.LAMBDAS).to(cfg.MODEL.DEVICE)
        self.lambdas.requires_grad = False

        self.inter_modules_dist = [
            self.dmc.bit_estimator_z_mv,
            self.dmc.mv_decoder,
            self.dmc.mv_encoder,
            self.dmc.mv_hyper_prior_decoder,
            self.dmc.mv_hyper_prior_encoder,
            self.dmc.mv_y_spatial_prior,
            self.dmc.mv_y_prior_fusion,
            self.dmc.optic_flow
        ]

        self.inter_modules_rate = [
            self.dmc.mv_y_q_basic,
            self.dmc.mv_y_q_scale
        ]

        self.recon_modules_rate = [
            self.dmc.y_q_basic,
            self.dmc.y_q_scale
        ]

        # -------- self.recon_modules_dist --------
        # bit_estimator_z                   +
        # context_fusion_net                +
        # contextual_decoder                +
        # contextual_encoder                +
        # contextual_hyper_prior_decoder    +
        # contextual_hyper_prior_encoder    +
        # recon_generation_net              +
        # y_prior_fusion                    +
        # y_spatial_prior                   +
        # temporal_prior_encoder            +
        # feature_adaptor_I                 +
        # feature_adaptor_P                 +
        # feature_extractor                 +

    def activate_modules_inter_dist(self):
        for p in self.dmc.parameters():
            p.requires_grad = False

        for m in self.inter_modules_dist:
            for p in m.parameters():
                p.requires_grad = True

    def activate_modules_inter_dist_rate(self):
        for p in self.dmc.parameters():
            p.requires_grad = False

        for m in self.inter_modules_dist:
            for p in m.parameters():
                p.requires_grad = True

        for p in self.inter_modules_rate:
            p.requires_grad = True

    def activate_modules_recon_dist(self):
        for p in self.dmc.parameters():
            p.requires_grad = True

        for m in self.inter_modules_dist:
            for p in m.parameters():
                p.requires_grad = False

        for p in self.recon_modules_rate + self.inter_modules_rate:
            p.requires_grad = False

    def activate_modules_recon_dist_rate(self):
        for p in self.dmc.parameters():
            p.requires_grad = True

        for m in self.inter_modules_dist:
            for p in m.parameters():
                p.requires_grad = False

        for p in self.inter_modules_rate:
            p.requires_grad = False

    def activate_modules_all(self):
        for p in self.dmc.parameters():
            p.requires_grad = True

    def train_single(self,
                     input: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     loss_dist_key: str,
                     loss_rate_keys: List[str],
                     p_frames: int):
        """
        Implements single stage training strategy (I -> P frames).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            optimizer : torch.optim.Optimizer
                Optimizer to train model.
            # TODO: add other params
        """
        n, t, c, h, w = input.shape
        assert 0 < p_frames < t
        assert self.lambdas.shape[0] == n

        result = {
            'rate': [],  # (N, (T - p_frames) * p_frames)
            'dist': [],  # (N, (T - p_frames) * p_frames)
            'loss': [],  # (1, (T - p_frames) * p_frames)
            'single_forwards': 0
        }

        labmdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        for t_i in range(0, t - p_frames):
            # Initialize I-frame
            dpb = {
                "ref_frame": input[:, t_i],
                "ref_feature": None,
                "ref_y": None,
                "ref_mv_y": None
            }

            # Forward P-frames
            for p_idx in range(0, p_frames):
                output = self.dmc.forward_one_frame(input[:, t_i + 1 + p_idx],
                                                    dpb,
                                                    self.dmc.mv_y_q_scale,
                                                    self.dmc.y_q_scale)
                for key in dpb.keys():
                    dpb[key] = output['dpb'][key].detach()

                # Calculate loss
                rate = torch.zeros_like(self.lambdas)
                for key in loss_rate_keys:
                    assert key in output
                    rate += output[key]

                assert loss_dist_key in output
                dist = output[loss_dist_key]

                loss = rate + dist * labmdas
                loss = torch.mean(loss)  # (N) -> (1)

                # Do optimization
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                result['rate'].append(rate)  # (N)
                result['dist'].append(dist)  # (N)
                result['loss'].append(loss)  # (1)
                result['single_forwards'] += 1

        result['rate'] = torch.stack(result['rate'], -1)  # (N, (T - p_frames) * p_frames)
        result['dist'] = torch.stack(result['dist'], -1)  # (N, (T - p_frames) * p_frames)
        result['loss'] = torch.stack(result['loss'], -1)  # ((T - p_frames) * p_frames)

        return result

    def train_cascade(self,
                      input: torch.Tensor,
                      optimizer: torch.optim.Optimizer,
                      loss_dist_key: str,
                      loss_rate_keys: List[str],
                      p_frames: int):
        """
        Implements cascaded loss training strategy (avg loss).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            optimizer : torch.optim.Optimizer
                Optimizer to train model.
            # TODO: add other params
        """
        n, t, c, h, w = input.shape
        assert 0 < p_frames < t
        assert self.lambdas.shape[0] == n

        result = {
            'rate': [],  # (N, T - p_frames)
            'dist': [],  # (N, T - p_frames)
            'loss': [],  # (1, T - p_frames)
            'single_forwards': 0
        }

        labmdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        for t_i in range(0, t - p_frames):
            # Initialize I-frame
            dpb = {
                "ref_frame": input[:, t_i],
                "ref_feature": None,
                "ref_y": None,
                "ref_mv_y": None
            }

            rate_list = []
            dist_list = []
            loss_list = []

            # Forward P-frames
            for p_idx in range(0, p_frames):
                output = self.dmc.forward_one_frame(input[:, t_i + 1 + p_idx],
                                                    dpb,
                                                    self.dmc.mv_y_q_scale,
                                                    self.dmc.y_q_scale)

                # Calculate loss
                rate = torch.zeros_like(self.lambdas)
                for key in loss_rate_keys:
                    assert key in output
                    rate += output[key]

                assert loss_dist_key in output
                dist = output[loss_dist_key]

                rate_list.append(rate)
                dist_list.append(dist)
                loss_list.append(rate + dist * labmdas)

            rate = torch.stack(rate_list, -1)  # (N, T)
            rate = torch.mean(rate, -1)  # (N, T) -> (N)

            dist = torch.stack(dist_list, -1)  # (N, T)
            dist = torch.mean(dist, -1)  # (N, T) -> (N)

            loss = torch.stack(loss_list, -1)  # (N, T)
            loss = torch.mean(loss, -1)  # (N, T) -> (N)
            loss = torch.mean(loss, -1)  # (N) -> (1)

            result['rate'].append(rate)  # (N)
            result['dist'].append(dist)  # (N)
            result['loss'].append(loss)  # (1)
            result['single_forwards'] += 1

            # Do optimization
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        result['rate'] = torch.stack(result['rate'], -1)  # (N, T - p_frames)
        result['dist'] = torch.stack(result['dist'], -1)  # (N, T - p_frames)
        result['loss'] = torch.stack(result['loss'], -1)  # (T - p_frames)

        return result

    def forward(self, input: torch.Tensor):
        """
        input: tensor with shape (N, T, C, H, W)
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

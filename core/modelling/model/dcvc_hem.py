from typing import List

import lpips
import torch
from torch import nn

from DCVC_HEM.src.models.video_model import DMC
from core.engine.losses import VGGPerceptualLoss, FasterRCNNFPNPerceptualLoss, YOLOV8PerceptualLoss, LPIPSPerceptualLoss


class DCVC_HEM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dmc = DMC(anchor_num=len(cfg.SOLVER.LAMBDAS))
        self.lambdas = torch.FloatTensor(cfg.SOLVER.LAMBDAS).cuda()
        self.lambdas.requires_grad = False
        self.pl_lambda = torch.tensor(cfg.SOLVER.PL_LAMBDA).cuda()
        self.pl_lambda.requires_grad = False
        self.dist_lambda = torch.tensor(cfg.SOLVER.DIST_LAMBDA).cuda()
        self.dist_lambda.requires_grad = False
        self.perceptual_loss = self.get_perceptual_loss(cfg)

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

    def get_perceptual_loss(this, cfg):
        if cfg.SOLVER.PL_MODEL == 'vgg':
            perceptual_loss = VGGPerceptualLoss()
        elif cfg.SOLVER.PL_MODEL == 'rcnn':
            perceptual_loss = FasterRCNNFPNPerceptualLoss()
        elif cfg.SOLVER.PL_MODEL == 'yolo':
            perceptual_loss = YOLOV8PerceptualLoss()
        elif cfg.SOLVER.PL_MODEL == 'lpips_linear':
            perceptual_loss = LPIPSPerceptualLoss(use_lpips=True, use_dropout=True)
        elif cfg.SOLVER.PL_MODEL == 'lpips_no_linear':
            perceptual_loss = LPIPSPerceptualLoss(use_lpips=False, use_dropout=False)
        else:
            raise SystemError('Invalid perceptual loss')

        perceptual_loss.disable_gradients()
        perceptual_loss.cuda()
        perceptual_loss.eval()
        return perceptual_loss

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

    def forward_single(self,
                       input: torch.Tensor,
                       target: torch.Tensor,
                       optimizer: torch.optim.Optimizer,
                       loss_dist_key: str,
                       loss_rate_keys: List[str],
                       p_frames: int,
                       perceptual_loss: bool,
                       is_train=True,
                       i_frame_net=None,
                       i_frame_q_scales=None):
        """
        Implements single stage training strategy (I -> P frames).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            target : tensor
                Target video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            optimizer : torch.optim.Optimizer
                Optimizer to train model.
            loss_dist_key: str
                Loss dist key for output dictionary
            loss_rate_keys: List[str]
                Loss rate keys for output dictionary
            p_frames: int
                Number of p-frames
            perceptual_loss: bool
                Perceptual loss usage
            is_train: bool
                Train or eval mode
        """
        n, t, c, h, w = input.shape
        assert 0 < p_frames < t
        assert self.lambdas.shape[0] == n

        result = {
            'rate': [],  # (N, (T - p_frames) * p_frames)
            'dist': [],  # (N, (T - p_frames) * p_frames)
            'p_dist': [],  # (N, (T - p_frames) * p_frames)
            'loss': [],  # (N, (T - p_frames) * p_frames)
            'loss_seq': [],  # (N, T - p_frames)
            'input_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
            'decod_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
            'single_forwards': 0
        }

        lambdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        for t_i in range(0, t - p_frames):
            # Initialize I-frame
            if i_frame_net is not None:
                out_images = []
                for i in range(n):
                    with torch.no_grad():
                        out_i = i_frame_net(input[i, t_i].unsqueeze(0), i_frame_q_scales[i])
                    out_images.append(out_i["x_hat"].squeeze(0))
                dpb = {
                    "ref_frame": torch.stack(out_images, 0),
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None
                }
            else:
                dpb = {
                    "ref_frame": input[:, t_i],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None
                }

            input_seqs = []
            decod_seqs = []
            input_seqs.append(target[:, t_i])
            decod_seqs.append(input[:, t_i])

            loss_list = []

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

                if perceptual_loss:
                    perceptual_dist = self.perceptual_loss(target[:, t_i + 1 + p_idx], output['dpb']['ref_frame'])
                else:
                    perceptual_dist = torch.zeros_like(self.lambdas)

                loss = rate + lambdas * (dist * self.dist_lambda + perceptual_dist * self.pl_lambda)
                loss_to_opt = torch.mean(loss)  # (N) -> (1)
                
                loss_list.append(loss)

                if is_train:
                    # Do optimization
                    optimizer.zero_grad()
                    loss_to_opt.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                result['rate'].append(rate)  # (N)
                result['dist'].append(dist)  # (N)
                result['p_dist'].append(perceptual_dist)  # (N)
                result['loss'].append(loss)  # (N)
                result['single_forwards'] += 1
                input_seqs.append(target[:, t_i + 1 + p_idx])
                decod_seqs.append(dpb["ref_frame"])

            loss_seq = torch.stack(loss_list, -1)  # (N, p_frames)
            loss_seq = torch.mean(loss_seq, -1)  # (N, p_frames) -> (N)

            result['loss_seq'].append(loss_seq)  # (N)
            result['input_seqs'].append(torch.stack(input_seqs, -1))  # (N, p_frames + 1)
            result['decod_seqs'].append(torch.stack(decod_seqs, -1))  # (N, p_frames + 1)

        result['rate'] = torch.stack(result['rate'], -1)  # (N, (T - p_frames) * p_frames)
        result['dist'] = torch.stack(result['dist'], -1)  # (N, (T - p_frames) * p_frames)
        result['p_dist'] = torch.stack(result['p_dist'], -1)  # (N, (T - p_frames) * p_frames)
        result['loss'] = torch.stack(result['loss'], -1)  # (N, (T - p_frames) * p_frames)
        result['loss_seq'] = torch.stack(result['loss_seq'], -1)  # (N, T - p_frames)
        result['input_seqs'] = torch.stack(result['input_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
        result['input_seqs'] = result['input_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)
        result['decod_seqs'] = torch.stack(result['decod_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
        result['decod_seqs'] = result['decod_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)

        return result

    def forward_single_multi(self,
                             input: torch.Tensor,
                             target: torch.Tensor,
                             loss_dist_key: str,
                             loss_rate_keys: List[str],
                             dpb,
                             perceptual_loss: bool):
        """
        Implements single stage training strategy (I -> P frames).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            target : tensor
                Target video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            loss_dist_key: str
                Loss dist key for output dictionary
            loss_rate_keys: List[str]
                Loss rate keys for output dictionary
            dpb: dict
                Decoded picture buffer
            perceptual_loss: bool
                Perceptual loss usage
        """
        n, c, h, w = input.shape
        assert self.lambdas.shape[0] == n

        lambdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        output = self.dmc.forward_one_frame(input,
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

        if perceptual_loss:
            perceptual_dist = self.perceptual_loss(target, output['dpb']['ref_frame'])
        else:
            perceptual_dist = torch.zeros_like(self.lambdas)

        loss = rate + lambdas * (dist * self.dist_lambda + perceptual_dist * self.pl_lambda)

        loss_to_opt = torch.mean(loss)  # (N) -> (1)
        result = {
            "rate": rate,  # (N)
            "dist": dist,  # (N)
            "p_dist": perceptual_dist,  # (N)
            "loss": loss,  # (N)
            "loss_to_opt": loss_to_opt,  # (1)
            "input_seqs": target,  # (N, p_frame)
            "decod_seqs": dpb["ref_frame"],  # (N, p_frame)
            "dpb": dpb
        }

        return result

    def forward_cascade(self,
                        input: torch.Tensor,
                        target: torch.Tensor,
                        optimizer: torch.optim.Optimizer,
                        loss_dist_key: str,
                        loss_rate_keys: List[str],
                        p_frames: int,
                        perceptual_loss: bool,
                        is_train=True,
                        i_frame_net=None,
                        i_frame_q_scales=None):
        """
        Implements cascaded loss training strategy (avg loss).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            target : tensor
                Target video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            optimizer : torch.optim.Optimizer
                Optimizer to train model.
            loss_dist_key: str
                Loss dist key for output dictionary
            loss_rate_keys: List[str]
                Loss rate keys for output dictionary
            p_frames: int
                Number of p-frames
            perceptual_loss: bool
                Perceptual loss usage
            is_train: bool
                Train or eval mode
        """
        n, t, c, h, w = input.shape
        assert 0 < p_frames < t
        assert self.lambdas.shape[0] == n

        result = {
            'rate': [],  # (N, T - p_frames)
            'dist': [],  # (N, T - p_frames)
            'p_dist': [],  # (N, T - p_frames)
            'loss': [],  # (N, T - p_frames)
            'loss_seq': [],  # (N, T - p_frames)
            'input_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
            'decod_seqs': [],  # (N, T - p_frames, p_frames + 1, C, H, W)
            'single_forwards': 0
        }

        lambdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        for t_i in range(0, t - p_frames):
            # Initialize I-frame
            if i_frame_net is not None:
                out_images = []
                for i in range(n):
                    with torch.no_grad():
                        out_i = i_frame_net(input[i, t_i].unsqueeze(0), i_frame_q_scales[i])
                    out_images.append(out_i["x_hat"].squeeze(0))
                dpb = {
                    "ref_frame": torch.stack(out_images, 0),
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None
                }
            else:
                dpb = {
                    "ref_frame": input[:, t_i],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None
                }

            input_seqs = []
            decod_seqs = []
            input_seqs.append(target[:, t_i])
            decod_seqs.append(input[:, t_i])

            rate_list = []
            dist_list = []
            p_dist_list = []
            loss_list = []

            # Forward P-frames
            for p_idx in range(0, p_frames):
                output = self.dmc.forward_one_frame(input[:, t_i + 1 + p_idx],
                                                    dpb,
                                                    self.dmc.mv_y_q_scale,
                                                    self.dmc.y_q_scale)
                dpb = output['dpb']

                # Calculate loss
                rate = torch.zeros_like(self.lambdas)
                for key in loss_rate_keys:
                    assert key in output
                    rate += output[key]

                assert loss_dist_key in output
                dist = output[loss_dist_key]

                if perceptual_loss:
                    perceptual_dist = self.perceptual_loss(target[:, t_i + 1 + p_idx], output['dpb']['ref_frame'])
                else:
                    perceptual_dist = torch.zeros_like(self.lambdas)

                rate_list.append(rate)
                dist_list.append(dist)
                p_dist_list.append(perceptual_dist)
                loss_list.append(rate + lambdas * (dist * self.dist_lambda + perceptual_dist * self.pl_lambda))
                input_seqs.append(target[:, t_i + 1 + p_idx])
                decod_seqs.append(dpb["ref_frame"])

            rate = torch.stack(rate_list, -1)  # (N, p_frames)
            rate = torch.mean(rate, -1)  # (N, p_frames) -> (N)

            dist = torch.stack(dist_list, -1)  # (N, p_frames)
            dist = torch.mean(dist, -1)  # (N, p_frames) -> (N)

            p_dist = torch.stack(p_dist_list, -1)  # (N, p_frames)
            p_dist = torch.mean(p_dist, -1)  # (N, p_frames) -> (N)

            loss = torch.stack(loss_list, -1)  # (N, p_frames)
            loss = torch.mean(loss, -1)  # (N, p_frames) -> (N)
            loss_to_opt = torch.mean(loss, -1)  # (N) -> (1)

            result['rate'].append(rate)  # (N)
            result['dist'].append(dist)  # (N)
            result['p_dist'].append(p_dist)  # (N)
            result['loss'].append(loss)  # (N)
            result['single_forwards'] += 1

            result['input_seqs'].append(torch.stack(input_seqs, -1))  # (N, p_frames + 1)
            result['decod_seqs'].append(torch.stack(decod_seqs, -1))  # (N, p_frames + 1)

            if is_train:
                # Do optimization
                optimizer.zero_grad()
                loss_to_opt.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        result['rate'] = torch.stack(result['rate'], -1)  # (N, T - p_frames)
        result['dist'] = torch.stack(result['dist'], -1)  # (N, T - p_frames)
        result['p_dist'] = torch.stack(result['p_dist'], -1)  # (N, T - p_frames)
        result['loss'] = torch.stack(result['loss'], -1)  # (N, T - p_frames)
        result['loss_seq'] = result['loss']  # (N, T - p_frames)
        result['input_seqs'] = torch.stack(result['input_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
        result['input_seqs'] = result['input_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)
        result['decod_seqs'] = torch.stack(result['decod_seqs'], -1)  # (N, C, H, W, p_frames + 1, T - p_frames)
        result['decod_seqs'] = result['decod_seqs'].permute(0, 5, 4, 1, 2, 3)  # (N, T - p_frames, p_frames + 1, C, H, W)

        return result

    def forward_cascade_multi(self,
                              input: torch.Tensor,
                              target: torch.Tensor,
                              loss_dist_key: str,
                              loss_rate_keys: List[str],
                              dpb,
                              p_frames: int,
                              t_i: int,
                              perceptual_loss: bool):
        """
        Implements cascaded loss training strategy (avg loss).
        See: https://arxiv.org/pdf/2111.13850v1.pdf

        Parameters:
            input : tensor
                Video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            target : tensor
                Target video sequences data with shape (N, T, C, H, W).
                N is equal to number of global quantization steps.
            loss_dist_key: str
                Loss dist key for output dictionary
            loss_rate_keys: List[str]
                Loss rate keys for output dictionary
            dpb: dict
                Decoded picture buffer
            p_frames: int
                Number of p-frames
            t_i: int
                Subsequence index
            perceptual_loss: bool
                Perceptual loss usage
        """
        n, t, c, h, w = input.shape
        assert self.lambdas.shape[0] == n

        lambdas = self.lambdas if len(loss_rate_keys) else torch.ones_like(self.lambdas)

        input_seqs = []
        decod_seqs = []
        input_seqs.append(target[:, t_i])
        decod_seqs.append(input[:, t_i])

        rate_list = []
        dist_list = []
        p_dist_list = []
        loss_list = []

        # Forward P-frames
        for p_idx in range(0, p_frames):
            output = self.dmc.forward_one_frame(input[:, t_i + 1 + p_idx],
                                                dpb,
                                                self.dmc.mv_y_q_scale,
                                                self.dmc.y_q_scale)
            dpb = output['dpb']

            # Calculate loss
            rate = torch.zeros_like(self.lambdas)
            for key in loss_rate_keys:
                assert key in output
                rate += output[key]

            assert loss_dist_key in output
            dist = output[loss_dist_key]

            if perceptual_loss:
                perceptual_dist = self.perceptual_loss(target[:, t_i + 1 + p_idx], output['dpb']['ref_frame'])
            else:
                perceptual_dist = torch.zeros_like(self.lambdas)

            rate_list.append(rate)
            dist_list.append(dist)
            p_dist_list.append(perceptual_dist)
            loss_list.append(rate + lambdas * (dist * self.dist_lambda + perceptual_dist * self.pl_lambda))
            input_seqs.append(target[:, t_i + 1 + p_idx])
            decod_seqs.append(dpb["ref_frame"])

        rate = torch.stack(rate_list, -1)  # (N, p_frames)
        rate = torch.mean(rate, -1)  # (N, p_frames) -> (N)

        dist = torch.stack(dist_list, -1)  # (N, p_frames)
        dist = torch.mean(dist, -1)  # (N, p_frames) -> (N)

        p_dist = torch.stack(p_dist_list, -1)  # (N, p_frames)
        p_dist = torch.mean(p_dist, -1)  # (N, p_frames) -> (N)

        loss = torch.stack(loss_list, -1)  # (N, p_frames)
        loss = torch.mean(loss, -1)  # (N, p_frames) -> (N)
        loss_to_opt = torch.mean(loss, -1)  # (N) -> (1)

        result = {
            "rate": rate,  # (N)
            "dist": dist,  # (N)
            "p_dist": p_dist,  # (N)
            "loss": loss,  # (N)
            "loss_to_opt": loss_to_opt,  # (1)
            "input_seqs": torch.stack(input_seqs, -1),  # (N, p_frames + 1)
            "decod_seqs": torch.stack(decod_seqs, -1),  # (N, p_frames + 1)
            "dpb": dpb
        }

        return result


    def forward_simple(self,
                       input: torch.Tensor,
                       dpb):
        n, t, c, h, w = input.shape
        assert self.lambdas.shape[0] == n

        out_dpb = []
        for i in range(n):
            output = self.dmc.forward_one_frame(input[i],
                                                dpb[i],
                                                self.dmc.mv_y_q_scale[i],
                                                self.dmc.y_q_scale[i])
            out_dpb.append(output['dpb'])

        return out_dpb

    def forward(self,
                forward_method: str,
                input: torch.Tensor,
                target: torch.Tensor = None,
                loss_dist_key: str = None,
                loss_rate_keys: List[str] = None,
                p_frames: int = None,
                perceptual_loss: bool = None,
                optimizer: torch.optim.Optimizer = None,
                is_train=True,
                dpb=None,
                t_i=None,
                i_frame_net=None,
                i_frame_q_scales=None):
        if forward_method == 'single':
            return self.forward_single(input, target, optimizer, loss_dist_key, loss_rate_keys, p_frames,
                                       perceptual_loss, is_train, i_frame_net, i_frame_q_scales)
        elif forward_method == 'single_multi':
            return self.forward_single_multi(input, target, loss_dist_key, loss_rate_keys, dpb, perceptual_loss)
        elif forward_method == 'cascade':
            return self.forward_cascade(input, target, optimizer, loss_dist_key, loss_rate_keys, p_frames,
                                        perceptual_loss, is_train, i_frame_net, i_frame_q_scales)
        elif forward_method == 'cascade_multi':
            return self.forward_cascade_multi(input, target, loss_dist_key, loss_rate_keys, dpb, p_frames, t_i,
                                              perceptual_loss)
        elif forward_method == 'forward_simple':
            return self.forward_simple(input, dpb)

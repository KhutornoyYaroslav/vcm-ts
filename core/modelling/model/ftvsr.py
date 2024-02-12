from torch import nn
from ..sr.ftvsr import FTVSRNet


class FTVSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = FTVSRNet(mid_channels=cfg.MODEL.MID_CHANNELS, num_blocks=cfg.MODEL.NUM_BLOCKS)

    def forward(self, lrs):
        return self.generator(lrs)

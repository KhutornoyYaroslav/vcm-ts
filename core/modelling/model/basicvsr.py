from torch import nn
from ..sr.basicvsr_net import BasicVSRNet


class BasicVSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = BasicVSRNet(mid_channels=cfg.MODEL.MID_CHANNELS, num_blocks=cfg.MODEL.NUM_BLOCKS)

    def forward(self, lrs):
        return self.generator(lrs)
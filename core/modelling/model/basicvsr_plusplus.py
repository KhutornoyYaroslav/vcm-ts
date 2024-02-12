from torch import nn
from ..sr.basicvsr_plusplus_net import BasicVSRPlusPlusNet


class BasicVSRPlusPlus(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = BasicVSRPlusPlusNet(mid_channels=cfg.MODEL.MID_CHANNELS, num_blocks=cfg.MODEL.NUM_BLOCKS)

    def forward(self, lrs):
        return self.generator(lrs)
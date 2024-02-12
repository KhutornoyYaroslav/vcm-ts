from .ftvsr import FTVSR
from .basicvsr_plusplus import BasicVSRPlusPlus
from .basicvsr import BasicVSR


_MODEL_ARCHITECTURES = {
    "FTVSR": FTVSR,
    "BasicVSR": BasicVSR,
    "BasicVSRPlusPlus": BasicVSRPlusPlus
}


def build_model(cfg):
    meta_arch = _MODEL_ARCHITECTURES[cfg.MODEL.ARCHITECTURE]
    return meta_arch(cfg)

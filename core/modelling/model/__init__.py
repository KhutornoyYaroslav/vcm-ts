from .dcvc_hem import DCVC_HEM


_MODEL_ARCHITECTURES = {
    "DCVC_HEM": DCVC_HEM,
}


def build_model(cfg):
    meta_arch = _MODEL_ARCHITECTURES[cfg.MODEL.ARCHITECTURE]
    return meta_arch(cfg)

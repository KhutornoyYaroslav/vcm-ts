from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# BASE LAYER
# ---------------------------------------------------------------------------- #
_CFG.BASE_LAYER = CN()

# ---------------------------------------------------------------------------- #
# BASE LAYER [DCVC_HEM]
# ---------------------------------------------------------------------------- #
_CFG.BASE_LAYER.DCVC_HEM = CN()
_CFG.BASE_LAYER.DCVC_HEM.I_FRAME_WEIGHTS = 'pretrained/acmmm2022_image_psnr.pth'
_CFG.BASE_LAYER.DCVC_HEM.P_FRAME_WEIGHTS = 'pretrained/acmmm2022_video_psnr.pth'
_CFG.BASE_LAYER.DCVC_HEM.ANCHOR_NUM = 4
_CFG.BASE_LAYER.DCVC_HEM.GOP = 32
_CFG.BASE_LAYER.DCVC_HEM.RATE_COUNT = 6
_CFG.BASE_LAYER.DCVC_HEM.QUALITY = 1
_CFG.BASE_LAYER.DCVC_HEM.WRITE_STREAM = True
_CFG.BASE_LAYER.DCVC_HEM.DEVICE = 'cuda'

# ---------------------------------------------------------------------------- #
# ENHANCEMENT LAYER
# ---------------------------------------------------------------------------- #
_CFG.ENHANCEMENT_LAYER = CN()

# ---------------------------------------------------------------------------- #
# ENHANCEMENT LAYER [DETECTORS]
# ---------------------------------------------------------------------------- #
_CFG.ENHANCEMENT_LAYER.DETECTORS = CN()

# ---------------------------------------------------------------------------- #
# ENHANCEMENT LAYER [DETECTORS FACES]
# ---------------------------------------------------------------------------- #
_CFG.ENHANCEMENT_LAYER.DETECTORS.FACES = CN()
_CFG.ENHANCEMENT_LAYER.DETECTORS.FACES.DEVICE = 'cuda'
_CFG.ENHANCEMENT_LAYER.DETECTORS.FACES.ENABLE = True
_CFG.ENHANCEMENT_LAYER.DETECTORS.FACES.PADDING = 10
_CFG.ENHANCEMENT_LAYER.DETECTORS.FACES.PROB = 0.9

# ---------------------------------------------------------------------------- #
# ENHANCEMENT LAYER [DETECTORS LIPLATES]
# ---------------------------------------------------------------------------- #
_CFG.ENHANCEMENT_LAYER.DETECTORS.LIPLATES = CN()
_CFG.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.DEVICE = 'cuda'
_CFG.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.ENABLE = True
_CFG.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PADDING = 10
_CFG.ENHANCEMENT_LAYER.DETECTORS.LIPLATES.PROB = 0.9

# ---------------------------------------------------------------------------- #
# ENHANCEMENT LAYER [h265]
# ---------------------------------------------------------------------------- #
_CFG.ENHANCEMENT_LAYER.H265 = CN()
_CFG.ENHANCEMENT_LAYER.H265.CRF = 25
_CFG.ENHANCEMENT_LAYER.H265.PIX_FMT = 'gbrp'
_CFG.ENHANCEMENT_LAYER.H265.PRESET = 'veryfast'

# ---------------------------------------------------------------------------- #
# COMPARE
# ---------------------------------------------------------------------------- #
_CFG.COMPARE = CN()

# ---------------------------------------------------------------------------- #
# COMPARE [h265]
# ---------------------------------------------------------------------------- #
_CFG.COMPARE.H265 = CN()
_CFG.COMPARE.H265.PIX_FMT = 'gbrp'
_CFG.COMPARE.H265.PRESET = 'veryfast'
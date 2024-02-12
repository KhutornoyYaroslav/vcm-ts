from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# CUTOUT LAYER
# ---------------------------------------------------------------------------- #
_CFG.CUTOUT_LAYER = CN()

# ---------------------------------------------------------------------------- #
# CUTOUT LAYER [DETECTORS]
# ---------------------------------------------------------------------------- #
_CFG.CUTOUT_LAYER.DETECTORS = CN()

# ---------------------------------------------------------------------------- #
# CUTOUT LAYER [DETECTORS FACES]
# ---------------------------------------------------------------------------- #
_CFG.CUTOUT_LAYER.DETECTORS.FACES = CN()
_CFG.CUTOUT_LAYER.DETECTORS.FACES.DEVICE = 'cuda'
_CFG.CUTOUT_LAYER.DETECTORS.FACES.ENABLE = True
_CFG.CUTOUT_LAYER.DETECTORS.FACES.PADDING = 10
_CFG.CUTOUT_LAYER.DETECTORS.FACES.PROB = 0.9

# ---------------------------------------------------------------------------- #
# CUTOUT LAYER [DETECTORS LIPLATES]
# ---------------------------------------------------------------------------- #
_CFG.CUTOUT_LAYER.DETECTORS.LIPLATES = CN()
_CFG.CUTOUT_LAYER.DETECTORS.LIPLATES.DEVICE = 'cuda'
_CFG.CUTOUT_LAYER.DETECTORS.LIPLATES.ENABLE = True
_CFG.CUTOUT_LAYER.DETECTORS.LIPLATES.PADDING = 10
_CFG.CUTOUT_LAYER.DETECTORS.LIPLATES.PROB = 0.9

# ---------------------------------------------------------------------------- #
# CUTOUT LAYER [h265]
# ---------------------------------------------------------------------------- #
_CFG.CUTOUT_LAYER.H265 = CN()
_CFG.CUTOUT_LAYER.H265.CRF = 15
_CFG.CUTOUT_LAYER.H265.PIX_FMT = 'gbrp'
_CFG.CUTOUT_LAYER.H265.PRESET = 'veryfast'

# ---------------------------------------------------------------------------- #
# VIDEO LAYER
# ---------------------------------------------------------------------------- #
_CFG.VIDEO_LAYER = CN()

# ---------------------------------------------------------------------------- #
# VIDEO LAYER [VSR MODEL]
# ---------------------------------------------------------------------------- #
_CFG.VIDEO_LAYER.VSR_MODEL = CN()
_CFG.VIDEO_LAYER.VSR_MODEL.ARCHITECTURE = 'FTVSR'
_CFG.VIDEO_LAYER.VSR_MODEL.DEVICE = 'cuda'
_CFG.VIDEO_LAYER.VSR_MODEL.CHUNK_SIZE = 10
_CFG.VIDEO_LAYER.VSR_MODEL.MID_CHANNELS = 64
_CFG.VIDEO_LAYER.VSR_MODEL.NUM_BLOCKS = 60
_CFG.VIDEO_LAYER.VSR_MODEL.TILE_PADDING = 16
_CFG.VIDEO_LAYER.VSR_MODEL.TILE_SIZE = 160
_CFG.VIDEO_LAYER.VSR_MODEL.WEIGHTS_PATH = 'pretrained/FTVSR_REDS.pth'

# ---------------------------------------------------------------------------- #
# VIDEO LAYER [h265]
# ---------------------------------------------------------------------------- #
_CFG.VIDEO_LAYER.H265 = CN()
_CFG.VIDEO_LAYER.H265.CRF = 25
_CFG.VIDEO_LAYER.H265.PIX_FMT = 'gbrp'
_CFG.VIDEO_LAYER.H265.PRESET = 'veryfast'

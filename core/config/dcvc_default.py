from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
# _CFG.MODEL = CN()
# _CFG.MODEL.ARCHITECTURE = 'FTVSR'
# _CFG.MODEL.DEVICE = "cuda"
# _CFG.MODEL.MID_CHANNELS = 64
# _CFG.MODEL.NUM_BLOCKS = 60
# _CFG.MODEL.SPYNET_FREEZE_FOR_EPOCHS = 1

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.MAKE_DIVISIBLE_BY = 8

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASET = CN()
_CFG.DATASET.TYPE = ''
_CFG.DATASET.TRAIN_ROOT_DIRS = []
_CFG.DATASET.TEST_ROOT_DIRS = []
_CFG.DATASET.SEQUENCE_LENGTH = 16
_CFG.DATASET.SEQUENCE_STRIDE = 1
_CFG.DATASET.SUBDIR_INPUTS = 'raw'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
# _CFG.DATA_LOADER = CN()
# _CFG.DATA_LOADER.NUM_WORKERS = 1
# _CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
# _CFG.SOLVER = CN()
# _CFG.SOLVER.MAX_EPOCH = 128
# _CFG.SOLVER.LR = 2e-4
# _CFG.SOLVER.LR_SPYNET = 1e-5
# _CFG.SOLVER.PERCEPTION_LOSS_WEIGHT = 0.01

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
# _CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
# _CFG.TENSORBOARD = CN()
# _CFG.TENSORBOARD.BEST_SAMPLES_NUM = 16
# _CFG.TENSORBOARD.WORST_SAMPLES_NUM = 16

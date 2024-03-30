from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CFG.MODEL = CN()
_CFG.MODEL.ARCHITECTURE = 'DCVC_HEM'
_CFG.MODEL.DEVICE = "cpu"
_CFG.MODEL.PRETRAINED_WEIGHTS = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.MAKE_DIVISIBLE_BY = 8
_CFG.INPUT.IMAGE_SIZE = (256, 256)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASET = CN()
_CFG.DATASET.TYPE = ''
_CFG.DATASET.TRAIN_ROOT_DIRS = []
_CFG.DATASET.TRAIN_SUBDIR_LISTS = []
_CFG.DATASET.TEST_ROOT_DIRS = []
_CFG.DATASET.TEST_SUBDIR_LISTS = []
_CFG.DATASET.TEST_OD_ROOT_DIRS = []
_CFG.DATASET.SEQUENCE_LENGTH = 16
_CFG.DATASET.SEQUENCE_STRIDE = 1
_CFG.DATASET.SUBDIR_INPUTS = 'raw'
_CFG.DATASET.METADATA_PATH = ''
_CFG.DATASET.OD_GOP_SIZE = 32
_CFG.DATASET.OD_STAGE = 5

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CFG.DATA_LOADER = CN()
_CFG.DATA_LOADER.NUM_WORKERS = 1
_CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CFG.SOLVER = CN()
_CFG.SOLVER.LAMBDAS = [85, 170, 380, 840]
_CFG.SOLVER.LR = 1e-4
_CFG.SOLVER.STAGES = [
    ['1', 'inter', 'single', 'me', 'none', '0.0001', '1'],
    ['1', 'inter', 'single', 'me', 'me', '0.0001', '3'],
    ['1', 'recon', 'single', 'rec', 'none', '0.0001', '3'],
    ['1', 'recon', 'single', 'rec', 'rec', '0.0001', '3'],
    ['1', 'all', 'single', 'rec', 'all', '0.0001', '6'],
    ['2', 'all', 'single', 'rec', 'all', '0.0001', '5'],
    ['4', 'all', 'single', 'rec', 'all', '0.0001', '3'],
    ['4', 'all', 'single', 'rec', 'all', '0.00001', '1'],
    ['4', 'all', 'cascade', 'rec', 'all', '0.00005', '2'],
    ['4', 'all', 'cascade', 'rec', 'all', '0.00001', '3']
  ]

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_CFG.TENSORBOARD = CN()
_CFG.TENSORBOARD.BEST_SAMPLES_NUM = 16
_CFG.TENSORBOARD.WORST_SAMPLES_NUM = 16

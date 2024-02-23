from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CFG.MODEL = CN()
_CFG.MODEL.ARCHITECTURE = 'DCVC_HEM'
_CFG.MODEL.DEVICE = "cpu"
# _CFG.MODEL.MID_CHANNELS = 64
# _CFG.MODEL.NUM_BLOCKS = 60
# _CFG.MODEL.SPYNET_FREEZE_FOR_EPOCHS = 1

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
_CFG.DATASET.TRAIN_LIST = []
_CFG.DATASET.TEST_ROOT_DIRS = []
_CFG.DATASET.TEST_LIST = []
_CFG.DATASET.SEQUENCE_LENGTH = 16
_CFG.DATASET.SEQUENCE_STRIDE = 1
_CFG.DATASET.SUBDIR_INPUTS = 'raw'

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
_CFG.SOLVER.STAGES = ['single', 'single', 'single', 'single', 'single', 'dual', 'multi', 'multi', 'multi', 'multi']
_CFG.SOLVER.PARTS = ['inter', 'inter', 'recon', 'recon', 'all', 'all', 'all', 'all', 'all', 'all']
_CFG.SOLVER.LOSS_TYPE = ['single', 'single', 'single', 'single', 'single', 'single', 'single', 'single', 'cascade', 'cascade']
_CFG.SOLVER.LOSS_DIST = ['me', 'me', 'rec', 'rec', 'rec', 'rec', 'rec', 'rec', 'rec', 'rec']
_CFG.SOLVER.LOSS_RATE = ['none', 'me', 'none', 'rec', 'all', 'all', 'all', 'all', 'all', 'all']
_CFG.SOLVER.LR = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00005, 0.00001]
_CFG.SOLVER.EPOCHS = [1, 3, 3, 3, 6, 5, 3, 1, 2, 3]
_CFG.SOLVER.MAX_EPOCH = 30

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

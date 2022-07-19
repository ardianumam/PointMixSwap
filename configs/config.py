from yacs.config import CfgNode as CN

_C = CN()
_C.ATTENTION = CN()
_C.ATTENTION.ENCODER_N_LAYER = 1
_C.ATTENTION.LINEAR_EMB = 512
_C.ATTENTION.N_HEAD = 1

_C.EXP = CN()
_C.EXP.DATASET = 'modelnet40'
_C.EXP.NAME = "exp_name"
_C.EXP.NUM_POINTS = 1024
_C.EXP.NUMPY_RANDOM_SEED = 7
_C.EXP.TORCH_RANDOM_SEED = 1
_C.EXP.UNIFORM_DATASET = False
_C.EXP.WORKING_DIR = "working_dir"

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.MIXUP_LEVEL = 'feature' #ops:  see README
_C.TRAIN.N_DIV = 3
_C.TRAIN.N_CLASSES = 40
_C.TRAIN.N_EPOCHS = 500
_C.TRAIN.OPT = "sgd"
_C.TRAIN.LR = 0.001
_C.TRAIN.MOMENTUM = 0.9 
_C.TRAIN.LABEL_SMOOTH = True

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 16

_C.DGCNN = CN()
_C.DGCNN.EMB_DIM = 1024
_C.DGCNN.K = 20
_C.DGCNN.DROPOUT = 0.5


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
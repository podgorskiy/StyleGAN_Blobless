from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = "results"

_C.DATASET = CN()
_C.DATASET.PATH = 'celeba/data_fold_%d_lod_%d.pkl'
_C.DATASET.FOLD_COUNT = 1

_C.MODEL = CN()

_C.MODEL.LAYER_COUNT = 6
_C.MODEL.START_CHANNEL_COUNT = 64
_C.MODEL.MAX_CHANNEL_COUNT = 512
_C.MODEL.LATENT_SPACE_SIZE = 256
_C.MODEL.DLATENT_AVG_BETA = 0.995
_C.MODEL.TRUNCATIOM_PSI = 0.7
_C.MODEL.STYLE_MIXING_PROB = 0.9

_C.TRAIN = CN()

_C.TRAIN.EPOCHS_PER_LOD = 15

_C.TRAIN.BASE_LEARNING_RATE = 0.00005 / 4
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = [90, 100]
_C.TRAIN.TRAIN_EPOCHS = 110

_C.TRAIN.ALPHA = 0.15
_C.TRAIN.M = 0.25
_C.TRAIN.BETTA = 0.02

_C.TRAIN.LOD_2_BATCH_8GPU = [512, 256, 128,   64,   32,    32]
_C.TRAIN.LOD_2_BATCH_4GPU = [512, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_2GPU = [256, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_1GPU = [128, 128, 128,   64,   32,    16]


def get_cfg_defaults():
    return _C.clone()

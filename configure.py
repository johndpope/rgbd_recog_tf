import os

# Directories
DIR_HOME        = os.path.expanduser('~')
DIR_CKPT        = 'checkpoints'
DIR_LST         = 'lists'
DIR_MODEL       = 'models'
DIR_SUMMARY     = 'summary'
DIR_DATA_RAW    = os.path.join(DIR_HOME, 'data', 'rgbd-dataset') #TODO: change accordingly
DIR_DATA        = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed') #TODO: change accordingly

# Lists
PTH_TRAIN_LST   = os.path.join(DIR_LST, 'train_full.lst')
PTH_EVAL_LST    = os.path.join(DIR_LST, 'eval_full.lst')
PTH_TEST_LST    = os.path.join(DIR_LST, 'test_full.lst')
PTH_DICT        = os.path.join(DIR_LST, 'dictionary.lst')

# Model
PTH_WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy') # AlexNet's pretrained model
PTH_MEAN_IMG    = os.path.join(DIR_MODEL, 'ilsvrc_2012_mean.npy') # mean image of imagenet dataset

# Classes
CLASSES         = open(PTH_DICT, 'r').read().splitlines()

# Parameters
IMG_S           = 227 # size of a square image

# Extensions
EXT_RGB         = '_crop.png'
EXT_D           = '_depthcrop.png'
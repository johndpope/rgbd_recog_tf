import os

# Directories
DIR_HOME        = os.path.expanduser('~')
DIR_CKPT        = 'checkpoints'
DIR_LST         = 'lists'
DIR_MODEL       = 'models'
DIR_SUMMARY     = 'summary'
DIR_DATA_RAW    = os.path.join(DIR_HOME, 'data', 'rgbd-dataset') #TODO: change accordingly
DIR_DATA        = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed') #TODO: change accordingly
DIR_DATA_4D     = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed4d') #TODO: change accordingly

# Lists
PTH_TRAIN_LST   = os.path.join(DIR_LST, 'train_full.lst')
PTH_EVAL_LST    = os.path.join(DIR_LST, 'eval_full.lst')
PTH_TEST_LST    = os.path.join(DIR_LST, 'test_full.lst')
PTH_DICT        = os.path.join(DIR_LST, 'dictionary.lst')

# Model
PTH_WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy') # AlexNet's pretrained model
PTH_MEAN_IMG    = os.path.join(DIR_MODEL, 'ilsvrc_2012_mean.npy') # mean image of imagenet dataset
PTH_RGB_MODEL   = os.path.join(DIR_MODEL, 'rgb_model.npy') # rgb model trained in phase 1
PTH_DEP_MODEL   = os.path.join(DIR_MODEL, 'dep_model.npy') # depth model trained in phase 1
PTH_FUS_MODEL   = os.path.join(DIR_MODEL, 'fus_model.npy') # fusion model trained in phase 2
PTH_RGBD_MEAN   = os.path.join(DIR_MODEL, 'rgbd_mean.npy')
PTH_RGBD_VAR    = os.path.join(DIR_MODEL, 'rgbd_var.npy')

# Classes
CLASSES         = open(PTH_DICT, 'r').read().splitlines()

# Parameters
IMG_S           = 227 # size of a square image

# Extensions of RGBD dataset
EXT_RGB         = '_crop.png'
EXT_D           = '_depthcrop.png'
EXT_4D          = '.npy'

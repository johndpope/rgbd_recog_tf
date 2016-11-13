import os

# Directories
DIR_HOME        = os.path.expanduser('~')
DIR_CKPT        = 'checkpoints'
DIR_LST         = 'lists'
DIR_MODEL       = 'models'
DIR_SUMMARY     = 'summary'
DIR_LOG         = 'logs'

DIR_DATA_RAW      = os.path.join(DIR_HOME, 'data', 'rgbd-dataset') #TODO: change accordingly
DIR_DATA          = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed') #TODO: change accordingly
DIR_DATA_4D       = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed4d') #TODO: change accordingly
DIR_DATA_EVAL_RAW = os.path.join(DIR_HOME, 'data', 'rgbd-dataset_eval')
DIR_DATA_EVAL     = os.path.join(DIR_HOME, 'data', 'rgbd-dataset_eval-processed')

# Lists
PTH_TRAIN_LST        = os.path.join(DIR_LST, 'train.lst')
PTH_EVAL_LST         = [os.path.join(DIR_LST, 'eval_'+str(trial+1)+'.lst') for trial in range(10)]
PTH_TESTINSTANCE_IDS = os.path.join(DIR_LST, 'testinstance_ids.txt')
#PTH_TEST_LST    = os.path.join(DIR_LST, 'test_full.lst')
PTH_DICT             = os.path.join(DIR_LST, 'dictionary.lst')

# Model
PTH_WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy') # AlexNet's pretrained model
PTH_MEAN_IMG    = os.path.join(DIR_MODEL, 'ilsvrc_2012_mean.npy') # mean image of imagenet dataset
PTH_RGB_MODEL   = os.path.join(DIR_MODEL, 'rgb_model.npy') # rgb model trained in phase 1
PTH_DEP_MODEL   = os.path.join(DIR_MODEL, 'dep_model.npy') # depth model trained in phase 1
PTH_FUS_MODEL   = os.path.join(DIR_MODEL, 'fus_model.npy') # fusion model trained in phase 2
PTH_RGBD_MEAN   = os.path.join(DIR_MODEL, 'rgbd_dataset_mean.npy')
PTH_RGBD_VAR    = os.path.join(DIR_MODEL, 'rgbd_dataset_var.npy')

# Classes
CLASSES         = open(PTH_DICT, 'r').read().splitlines()

# Parameters
N_TRIALS        = 10
IMG_S           = 227 # size of a square image
DEP_MIN         = 0.0
DEP_MAX         = 4000.0 # because of Kinect

# Extensions of RGBD dataset
EXT_RGB         = '_crop.png'
EXT_D           = '_depthcrop.png'
EXT_4D          = '.npy'

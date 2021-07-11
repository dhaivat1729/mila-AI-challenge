# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Borrowed as it is from detectron2! This is a base config,
# make change in main file.
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()



_C.CONFIG = CN()

"""
models = ['UNet',
		'fcn_resnet101',
		'fcn_resnet50',
		'deeplabv3_resnet50', 
		'deeplabv3_resnet101', 
		'deeplabv3_mobilenet_v3_large',
		'lraspp_mobilenet_v3_large',
		'deeplabv3_mobilenet_v3_small',
		'lraspp_mobilenet_v3_small']

"""

## Default model!
_C.CONFIG.MODEL_NAME = 'deeplabv3_mobilenet_v3_small'

## seed, important for reproducibility!
_C.CONFIG.SEED = 42

## if we are loading a pretrained model or not
_C.CONFIG.PRETRAINED = False

## dataset directory
_C.CONFIG.DATASET_DIR = '/network/tmp1/bhattdha/segmentation_project/'

## NUMBER OF channels in the input
## currently, only 3 is supported
_C.CONFIG.NUM_CHANNELS = 3


## NUMBER OF channels in the input
"""
This is a simple binary classifier.
SO no background class. 
"""
_C.CONFIG.NUM_CLASSES = 1

## provide a path to the final directory
## change this in the file correponding to
## the model being trained
_C.CONFIG.ROOT_DIR = 'logs'

## batchsize for trainloader/validationloader
_C.CONFIG.BATCHSIZE = 128

## train/val/test proportion
_C.CONFIG.TRAIN_PROP = 0.75
_C.CONFIG.VAL_PROP = 0.25
_C.CONFIG.TEST_PROP = 0.0

## train images path
## change it in the main script
_C.CONFIG.INPUT_PATH = '/network/tmp1/bhattdha/segmentation_project/train/img/'
_C.CONFIG.MASK_PATH = '/network/tmp1/bhattdha/segmentation_project/train/mask/'

## whether to shuffle the dataset or not
_C.CONFIG.DATA_SHUFFLE = True

## desired shape of input/label image
## when none, it loads default size!
## could be 'default or list --> [240,360]'
_C.CONFIG.INPUT_SIZE = 'default'

## where to save the model and tensorboard logs
_C.CONFIG.SAVING_FOLDERS_NAME = 'logs/model_name'

## weight of a minority class(to tackle class imbalance)
## this is only used in weighted BCE loss
_C.CONFIG.MINORITY_CLASS_WEIGHT = 20

## training epochs
_C.CONFIG.TRAIN_EPOCHS = 30

## early stopping patience
_C.CONFIG.PATIENCE = 10

## loss functions!
## It could be anything in ['WeightedBCE', 'IoULoss']
_C.CONFIG.LOSS_TYPE = 'WeightedBCE'
_C.CONFIG.JACCARD_LOSS_WEIGHT = 20

## L2 weight regularization
_C.CONFIG.L2_PENALTY = 0.0

## learning rate
_C.CONFIG.LR = 0.1

## learning rate scheduling
_C.CONFIG.LR_SCHEDULER_MS = [12, 20, 25]
_C.CONFIG.LR_SCHEDULE_DECAY = 0.1

## torch device
_C.CONFIG.DEVICE = ''

## freeze backbone
_C.CONFIG.FREEZE_BACKBONE = False

## validation metric
## could be from ['IoU', 'val_loss']
_C.CONFIG.METRIC = 'val_loss' 
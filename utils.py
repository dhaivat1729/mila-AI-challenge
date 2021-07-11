import numpy as np
import os
import torch
from src.config import get_cfg
from fvcore.common.file_io import PathManager


## this is temporary, must be fixed!
pretrained_flags = {'fcn_resnet101':True, 
				'fcn_resnet50':True, 
				'deeplabv3_resnet50': True, 
				'deeplabv3_resnet101': True, 
				'deeplabv3_mobilenet_v3_large': True,
				'lraspp_mobilenet_v3_large': True,
				'deeplabv3_mobilenet_v3_small': False,
				'lraspp_mobilenet_v3_small': False}	## these two models are not in the original repo, so there are no pretrained weights!


def setup(args):

	dataset_path, model_name, model_version = args['dataset_directory'], args['model_to_train'], args['model_version']

	## check cuda availability!
	assert torch.cuda.is_available(), "Code is not supported for CPU."
	device = torch.device('cuda')
	
	## get base config
	cfg = get_cfg()

	## modify the config
	cfg.CONFIG.MODEL_NAME = model_name
	cfg.CONFIG.DATASET_DIR = dataset_path

	## paths for train data, this is for MILA dataset, it's different for other datasets!
	cfg.CONFIG.INPUT_PATH = os.path.join(dataset_path, 'train', 'img') ## all the RGB images should be here
	cfg.CONFIG.MASK_PATH = os.path.join(dataset_path, 'train', 'mask') ## all the groundtruth masks should be here!

	## freeze the backbone if weights are pretrained
	# if pretrained_flags[model_name]:
	cfg.CONFIG.FREEZE_BACKBONE = False

	## Output directory
	## You can override this to any arbitrary path!
	cfg.CONFIG.SAVING_FOLDERS_NAME = os.path.join(dataset_path, 'logs', cfg.CONFIG.MODEL_NAME + '_' + model_version)

	## make directories
	os.makedirs(cfg.CONFIG.SAVING_FOLDERS_NAME, exist_ok = True)

	## L2 penalty(make it command line friendly)
	cfg.CONFIG.L2_PENALTY = 0.0

	## IoU as validation metric
	cfg.CONFIG.METRIC = 'IoU'

	## Loss function to be used
	cfg.CONFIG.LOSS_TYPE = "IoULoss"

	## freeze the config.
	cfg.freeze()

	## save the config
	## save as object and yaml file. YAML file can be used to merge it with existing config node
	torch.save({'cfg': cfg}, os.path.join(cfg.CONFIG.SAVING_FOLDERS_NAME, 'config.final'))
	with PathManager.open(os.path.join(cfg.CONFIG.SAVING_FOLDERS_NAME, 'config.yaml'), "w") as f:
		f.write(cfg.dump())

	return cfg




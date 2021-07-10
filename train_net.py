import sys
import os
from src import LoadDataset, Trainer
import torch
import torchvision
from torchsummary import summary
import random
import numpy as np
import argparse
from utils import setup, pretrained_flags

## currently, these segmentation models are supported, more to come!
from src.models.segmentation import (fcn_resnet101, 
									fcn_resnet50, 
									deeplabv3_resnet50, 
									deeplabv3_resnet101, 
									deeplabv3_mobilenet_v3_large, 
									lraspp_mobilenet_v3_large, 
									deeplabv3_mobilenet_v3_small, 
									lraspp_mobilenet_v3_small)

## extract the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-dataset_path", "--dataset_directory", required = True, help="Directory where the dataset is located") 
ap.add_argument("-model_name", "--model_to_train", required = True, help="which model to train")  
ap.add_argument("-model_ver", "--model_version", required = True, help="v1/v2/v3 - Which model is being trained.")  

args = vars(ap.parse_args())

models_supported = ['fcn_resnet101','fcn_resnet50', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large',
					'lraspp_mobilenet_v3_large','deeplabv3_mobilenet_v3_small','lraspp_mobilenet_v3_small']


## ensure that we have correct model
model_name = args['model_to_train']
assert model_name in models_supported, "Unknown model : {}".format(model_name)

cfg = setup(args)

### set all seeds for a reproducible behaviour!
torch.manual_seed(cfg.CONFIG.SEED)
torch.cuda.manual_seed_all(cfg.CONFIG.SEED)
np.random.seed(cfg.CONFIG.SEED)
random.seed(cfg.CONFIG.SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## model dictionary! Bad, could get better!
model_names_dict = {'fcn_resnet101':fcn_resnet101, 
	'fcn_resnet50':fcn_resnet50, 
	'deeplabv3_resnet50': deeplabv3_resnet50, 
	'deeplabv3_resnet101': deeplabv3_resnet101, 
	'deeplabv3_mobilenet_v3_large': deeplabv3_mobilenet_v3_large,
	'lraspp_mobilenet_v3_large': lraspp_mobilenet_v3_large,
	'deeplabv3_mobilenet_v3_small': deeplabv3_mobilenet_v3_small,
	'lraspp_mobilenet_v3_small': lraspp_mobilenet_v3_small}


## dataloaders!
train_dataset = LoadDataset(data_type = "train", cfg = cfg)
val_dataset = LoadDataset(data_type = "validation", cfg = cfg)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.CONFIG.BATCHSIZE, shuffle=cfg.CONFIG.DATA_SHUFFLE)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = cfg.CONFIG.BATCHSIZE, shuffle=cfg.CONFIG.DATA_SHUFFLE)

net = model_names_dict[model_name](pretrained=pretrained_flags[model_name], progress = 1, num_classes = cfg.CONFIG.NUM_CLASSES)

### model summary to be printed
# net = net.cuda()
# summary(net, (3, 240, 320))

### trainer object
trainer = Trainer(model = net, train_loader = train_loader, validation_loader = val_loader, cfg = cfg)

## freeze backbone 
if cfg.CONFIG.FREEZE_BACKBONE:
	for name, p in trainer.model.named_parameters():
		if 'backbone' in name:
			p.requires_grad = False

### Let's train
trainer.train()


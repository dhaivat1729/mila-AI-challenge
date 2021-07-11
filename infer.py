import sys
from src.dataloader import rgb2array
import os
import torch
import glob
import random
import numpy as np
import collections
from PIL import Image
from src.models.segmentation import (fcn_resnet101, 
									fcn_resnet50, 
									deeplabv3_resnet50, 
									deeplabv3_resnet101, 
									deeplabv3_mobilenet_v3_large, 
									lraspp_mobilenet_v3_large, 
									deeplabv3_mobilenet_v3_small, 
									lraspp_mobilenet_v3_small)
from src.config import get_cfg
from tqdm import tqdm
from utils import setup, pretrained_flags

## let's get the paths
input_path, output_path = sys.argv[1], sys.argv[2]

## check cuda availability!
assert torch.cuda.is_available(), "Codebase is not supported for CPU yet."

device = torch.device('cuda')

## this is where all the images are going to be stored!
cfg = get_cfg()

## this is the best model! Many other models are there to be tested too.
model_directory = '/network/tmp1/bhattdha/segmentation_project/logs/deeplabv3_resnet101_v1_jaccard_training'

## loading the original config
cfg.merge_from_file(os.path.join(model_directory, 'config.yaml'))

### set all seeds for a reproducible behaviour!
torch.manual_seed(cfg.CONFIG.SEED)
torch.cuda.manual_seed_all(cfg.CONFIG.SEED)
np.random.seed(cfg.CONFIG.SEED)
random.seed(cfg.CONFIG.SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model_name = cfg.CONFIG.MODEL_NAME

## model dictionary! 
model_names_dict = {'fcn_resnet101':fcn_resnet101, 
	'fcn_resnet50':fcn_resnet50, 
	'deeplabv3_resnet50': deeplabv3_resnet50, 
	'deeplabv3_resnet101': deeplabv3_resnet101, 
	'deeplabv3_mobilenet_v3_large': deeplabv3_mobilenet_v3_large,
	'lraspp_mobilenet_v3_large': lraspp_mobilenet_v3_large,
	'deeplabv3_mobilenet_v3_small': deeplabv3_mobilenet_v3_small,
	'lraspp_mobilenet_v3_small': lraspp_mobilenet_v3_small}


## get image list
image_list = glob.glob(os.path.join(input_path, '*'))

net = model_names_dict[model_name](pretrained=pretrained_flags[model_name], progress = 1, num_classes = cfg.CONFIG.NUM_CLASSES)

## loading the model
model_path = os.path.join(model_directory, 'final_model.pth')
assert os.path.isfile(model_path), "No such model at path {}".format(model_path)

## loading the state dict of trained model!
net.load_state_dict(torch.load(model_path))

## make directory to store test results
os.makedirs(output_path, exist_ok=True)

## tqdm bar
progress_bar = tqdm(total=len(image_list))

print("Running model on {} images from {} directory.".format(len(image_list), input_path))

## test model for all images
for i, name in enumerate(image_list):

	## update tqdm bar
	progress_bar.update()

	## this can get better!
	net = net.cuda()
	net.eval()

	## getting image name, without extension
	img_name = os.path.basename(name).split('.')[0]

	## image as numpy array
	img_org = rgb2array(name, desired_size = cfg.CONFIG.INPUT_SIZE)
		
	## basic normalization!
	img = (img_org - img_org.min()) / (img_org.max() - img_org.min())

	img = torch.from_numpy(img)
	img = img.permute(2,0,1).cuda()
	img = img[None,:,:,:]

	## forward pass
	output = net(img)
	if isinstance(output, collections.OrderedDict):
		output = output['out']

	## get the output mask
	output = torch.sigmoid(output).cpu().clone().detach().numpy().squeeze()*255
	output[output > 128] = 255
	output[output <= 128] = 0

	bmp_image = Image.fromarray(output.astype(np.uint8))
	bmp_image.save(os.path.join(output_path, img_name + '.bmp'))




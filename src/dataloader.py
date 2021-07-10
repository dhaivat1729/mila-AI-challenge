from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from collections import OrderedDict
from PIL import Image
import glob

def rgb2array(data_path,
              desired_size=None,
              expand=False,
              hwc=True):

    """Loads a 24-bit RGB image as a 3D numpy array."""

    img = Image.open(data_path).convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.float32) 
    if not hwc:
        x = np.transpose(x, [2, 0, 1])
    if expand:
        x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    return x

def label2array(data_path, desired_size=None, hwc=True, show=False):

    """Loads a bit map image."""

    img = Image.open(data_path)
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.int64)[..., 1]
    if show:
        plt.imshow(x, norm=MidpointNorm(0, 255, 1), interpolation='nearest')
    x = np.expand_dims(x, axis=0)
    if hwc:
        x = np.transpose(x, [1, 2, 0])
    x[x < 128] = 0.
    x[x >= 128] = 1.
    return x


class LoadDataset(Dataset):
    """Load MILA segmentation dataset."""

    def __init__(self, data_type = "validation", cfg = None):
        """
        Args:
            data_type (string): whether training or validation
            cfg(cfg noe): cfg node        
        """
        
        assert os.path.exists(root_path), "No such path exists!"

        ## config must be available
        assert cfg is not None, "cfg node can't be None"


        ## use this config as needed
        self.cfg = cfg


        ### path to input images and masks
        self.rgb_imgs_path = cfg.CONFIG.INPUT_PATH
        self.mask_imgs_path = cfg.CONFIG.MASK_PATH

        # train_imgs_path = os.path.join(self.root_path, 'train', 'img')
        # mask_imgs_path = os.path.join(self.root_path, 'train', 'mask')

        self.image_paths = [os.path.basename(x) for x in glob.glob(self.rgb_imgs_path + '*.jpg')]
        self.image_paths.sort() ## sorting to ensure that we have comparable models

        self.dataset_size = len(self.train_image_names)

        ## let's do train/test/validation split
        self.train_proportion = cfg.CONFIG.TRAIN_PROP
        self.val_proportion = cfg.CONFIG.VAL_PROP
        self.test_proportion = cfg.CONFIG.TEST_PROP

        ## train/val/test image paths
        self.train_images = self.image_paths[:int(self.train_proportion*self.dataset_size)]
        self.val_images = self.image_paths[int(self.train_proportion*self.dataset_size):int( (self.train_proportion + self.val_proportion) * self.dataset_size)]
        self.test_images = self.image_paths[-int(self.test_proportion * self.dataset_size):]

        ## final list with image names
        if data_type == "train":
            self.images_list = self.train_images
        elif data_type == "validation":
            self.image_list = self.val_images
        elif data_type == "test":
            self.image_list = self.test_images
        else:
            print("data_type can only be 'train', 'validation' or 'test'.")
            sys.exit(0)

 

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.rgb_imgs_path, self.train_images[idx])
        mask_path = os.path.join(self.mask_imgs_path, self.train_images[idx][:-4] + '.bmp') ## loading groundtruth corresponding to an image
        
        ## change input image size based on config!
        if self.cfg.CONFIG.INPUT_SIZE == 'default':
            desired_size = None
        else:
            desired_size = self.cfg.CONFIG.INPUT_SIZE

        img, label = rgb2array(img_path, desired_size = desired_size), label2array(label_path, desired_size = desired_size)
        
        ## basic normalization! This can get better. Come back to it later
        img = (img - img.min()) / (img.max() - img.min())

        ## prepare the data.
        final_image = torch.from_numpy(img)
        final_image = final_image.permute(2,0,1)
        label = torch.from_numpy(label).permute(2,0,1)
        
        ## data!
        sample = {'image': final_image, 'label': label}

        return sample


# # """
# # test the DataLoader

# # """
# train_dataset = LoadDataset(root_path = '', data_type = "validation")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle=True)
# data_loader_iter = iter(train_loader)
# x = next(data_loader_iter)

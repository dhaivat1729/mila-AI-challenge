import torch
import torchvision
from .early_stopping import EarlyStopping
import math
from tqdm import tqdm
import os
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import collections

class Trainer(object):
	"""
	Trainer object: 
		Here we implement basic form of training logic using early_stopping,
		Here we employ early stopping on precision of a foreground class, which maybe most critical! 

	"""
	def __init__(self, model = None, train_loader = None, validation_loader = None, cfg = None):
		super(Trainer, self).__init__() 

		# train loss/validation loss/average train loss/average validation loss
		self.train_losses, self.valid_losses, self.avg_train_losses, self.avg_valid_losses = [], [], [], []
		
		## let's keep track of precision, model with highest precision should be stored
		self.val_precision = []
		self.val_IoUs = []

		## this is only needed if we want to initialize the model from scratch
		# self.model = self.weight_init(model)
		self.model = model

		## config node
		self.cfg = cfg

		assert self.cfg is not None, "cfg node can't be None"

		## training epochs
		self.n_epochs = self.cfg.CONFIG.TRAIN_EPOCHS

		## minority class weight
		self.minority_cl_wt = self.cfg.CONFIG.MINORITY_CLASS_WEIGHT

		## Optimizer! TODO: Made this config friendly.
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.CONFIG.LR, weight_decay = self.cfg.CONFIG.L2_PENALTY) 
		
		## Learning rate scheduler
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.cfg.CONFIG.LR_SCHEDULER_MS, gamma = self.cfg.CONFIG.LR_SCHEDULE_DECAY)

		## not being used for now!
		self.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
		
		## train/validation loader
		self.train_loader = train_loader
		self.val_loader = validation_loader
		self.model_name = self.cfg.CONFIG.MODEL_NAME	## when you will have config, this will be changed accordingly!
		
		# self.saving_folder_name = saving_folder_name
		# if not(os.path.isdir(self.saving_folder_name)):
		# 	print('Making dir {}'.format(self.saving_folder_name))
		# 	os.makedirs(saving_folder_name)

		self.output_dir = self.cfg.CONFIG.SAVING_FOLDERS_NAME
		
		self.eps = 1e-8

		## in future, this is to resume training. 
		self.iter_tr = 0
		self.iter_test = 0

		## Push everything to tensorboard
		self.writer = SummaryWriter(self.output_dir)

		## final model to be stored
		self.model_path = os.path.join(self.output_dir, 'final_model.pth')

	def weight_init(self, model):

		"""
			Model weight initialization!
			This is very primitive, can get better.
		"""
		
		## TODO: Support config based weight initialization
		for name, param in model.named_parameters():
			torch.nn.init.normal_(param)

		return model

	def IoU(self, pred_mask, gt_mask, label = 1):
		""" 
		A function that compute the Intersection over Union (IoU)
		for the pixels with a given label between the prediction and the mask!
		Borrowed from the evaluation file
		"""

		assert pred_mask.shape == gt_mask.shape
		pred_label = (pred_mask == label).type(torch.int)
		mask_label = (gt_mask == label).type(torch.int)

		intersection = pred_label * mask_label
		union = (pred_label + mask_label - intersection)

		iscore = intersection.sum()
		uscore = union.sum()

		assert uscore != 0, 'the label {} is not present in the pred and the mask'.format(label)

		return iscore / uscore



	def weightBCEwithlogits(self, label, output):
		
		"""
			Weighted binary cross entropy loss
		"""
		
		label = label.flatten()
		output = output.flatten()

		## let's calculate weights
		ones = (label == 1).sum()
		zers = (label == 0).sum()

		## if we compute weight based on foreground/background pixels. Not very stable!
		w0 = (ones+zers) / (2 * zers)
		w1 = (ones+zers) / (2 * ones)	
		
		## converting logits to probability for binary classification
		output = torch.sigmoid(output)

		## Weighted binary cross entropy loss
		# loss = -(w1 * label * torch.log(output + self.eps) + w0 * (1 - label) * torch.log(1 - output + self.eps))
		loss = -(self.minority_cl_wt * label * torch.log(output + self.eps) + (1 - label) * torch.log(1 - output + self.eps))
		return loss.mean()



	def train(self):

		"""
			train and validate results!
		"""

		self.progress_bar =  tqdm(total=self.n_epochs * len(self.train_loader) + self.n_epochs * len(self.val_loader))
		self.early_stopping = EarlyStopping(patience=self.cfg.CONFIG.PATIENCE, verbose=True, path=self.model_path)

		for epoch in range(1, self.n_epochs + 1):
			
			"""
				Run one epoch of training and validation
			"""
			### train the model for one epoch
			###################
			# train the model #
			###################
			self.model = self.model.cuda()
			self.model.train() # prep model for training
			for batch, data in enumerate(self.train_loader, 1):
				
				self.progress_bar.update()

				## input
				image, label = data['image'].cuda(), data['label'].cuda()
				
				# clear the gradients of all optimized variables
				self.optimizer.zero_grad()
				
				# forward pass: compute predicted outputs by passing inputs to the model
				output = self.model(image)
				if isinstance(output, collections.OrderedDict):
					output = output['out']
				
				# calculate the loss
				loss = self.weightBCEwithlogits(label.type('torch.cuda.FloatTensor'), output) ## TODO: Make this better. This is hacky!
				
				# backward pass
				loss.backward()
				
				# perform a single optimization step (parameter update)
				self.optimizer.step()
				
				# record training loss
				self.train_losses.append(loss.item())
				
				## put it in tensorboard log!
				self.writer.add_scalar('Loss/train', loss.item(), self.iter_tr)
				self.iter_tr += 1

			### validate the model over one epoch
			######################    
			# validate the model #
			######################
			self.model.eval() # prep model for evaluation
			with torch.no_grad():
				
				for data in self.val_loader:
				
					self.progress_bar.update()

					# forward pass: compute predicted outputs by passing inputs to the model
					image, label = data['image'].cuda(), data['label'].cuda()
					
					# forward pass: compute predicted outputs by passing inputs to the model
					output = self.model(image)
					if isinstance(output, collections.OrderedDict):
						output = output['out']
					
					# calculate the loss
					loss = self.weightBCEwithlogits(label.type('torch.cuda.FloatTensor'), output) ## fix this datatype thing, it's ugly!
					
					# record validation loss
					self.valid_losses.append(loss.item())
					
					## let's compute the statistics too
					# label_numpy = label.cpu().clone().numpy()
					output_mask = torch.sigmoid(output)
					output_mask[output_mask > 0.5] = 1
					output_mask[output_mask <= 0.5] = 0

					## let's compute IoU! Final evaluation is to be done with IoU
					IoU_val = self.IoU(pred_mask = output_mask, gt_mask = label).item()
					self.val_IoUs.append(IoU_val)

					
					# self.val_precision.append(precision)
					self.writer.add_scalar('Loss/val', loss.item(), self.iter_test)
					self.writer.add_scalar('IoU/val', IoU_val, self.iter_test)
					self.iter_test +=1

			# print training/validation statistics 
			# calculate average loss over an epoch
			train_loss = np.average(self.train_losses)
			valid_loss = np.average(self.valid_losses)
			valid_IoU = np.average(self.val)
			# val_prec = np.average(self.val_precision)
			self.avg_train_losses.append(train_loss)
			self.avg_valid_losses.append(valid_loss)
			

			epoch_len = len(str(self.n_epochs))
			
			print_msg = (f'[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] ' +
						 f'train_loss: {train_loss:.5f} ' +
						 f'valid_loss: {valid_loss:.5f}')
			
			print(print_msg)
			
			# clear lists to track next epoch
			self.train_losses = []
			self.valid_losses = []
			self.val_precision = []
			self.val_IoUs = []
				
			# decide how to early stop based on validation metric from config
			if self.cfg.CONFIG.METRIC == 'val_loss':
				val_metric = valid_loss
			elif self.cfg.CONFIG.METRIC == 'IoU':
				val_metric = -valid_IoU ## because IoU needs to be high!

			# early_stopping needs the validation metric to decrease
			self.early_stopping(valid_IoU, self.model)
			
			### break if we need to early stop!
			if self.early_stopping.early_stop:
				print("Early stopping")
				break
			
			"""	
				epoch is over! Step for the scheduler
			"""			
			self.scheduler.step()

		# load the last checkpoint with the best model
		self.model.load_state_dict(torch.load(self.model_path))
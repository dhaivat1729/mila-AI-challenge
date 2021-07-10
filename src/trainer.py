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
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.CONFIG.LR) 
		
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

		progress_bar =  tqdm(total=self.n_epochs * len(self.train_loader) + self.n_epochs * len(self.val_loader))
		n_iterations = 0
		first_epoch = 0
		
		early_stopping = EarlyStopping(patience=self.cfg.CONFIG.PATIENCE, verbose=True, path=self.model_path)

		for epoch in range(1, self.n_epochs + 1):
			

			###################
			# train the model #
			###################
			self.model = self.model.cuda()
			self.model.train() # prep model for training
			for batch, data in enumerate(self.train_loader, 1):
				
				progress_bar.update()

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


			######################    
			# validate the model #
			######################
			self.model.eval() # prep model for evaluation
			with torch.no_grad():
				
				for data in self.val_loader:
				
					progress_bar.update()

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
					label_numpy = label.cpu().clone().numpy()
					output_numpy = torch.sigmoid(output).cpu().clone().detach().numpy()
					output_numpy[output_numpy > 0.5] = 1
					output_numpy[output_numpy <= 0.5] = 0

					# # results = classification_report(list(label_numpy.flatten().astype(int)), list(output_numpy.flatten().astype(int)), labels=[0,1], output_dict=True)
					# conf_mat = confusion_matrix(list(label_numpy.flatten().astype(int)), list(output_numpy.flatten().astype(int)))

					# tn, fp, fn, tp = conf_mat.ravel()
					
					# precision = tp / (tp + fp)

					# print("tn, fp, fn, tp ", conf_mat.ravel())
					
					# print("Precision of foreground class is: ", precision)
					
					# self.val_precision.append(precision)
					self.writer.add_scalar('Loss/val', loss.item(), self.iter_test)
					# self.writer.add_scalar('Precision/val', precision, self.iter_test)
					self.iter_test +=1
		
			# print training/validation statistics 
			# calculate average loss over an epoch
			train_loss = np.average(self.train_losses)
			valid_loss = np.average(self.valid_losses)
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
			
			# early_stopping needs the validation precision to check if it has increased, 
			# and if it has, it will make a checkpoint of the current model
			early_stopping(valid_loss, self.model)
			
			if early_stopping.early_stop:
				print("Early stopping")
				break
			
		# load the last checkpoint with the best model
		self.model.load_state_dict(torch.load(self.model_path))
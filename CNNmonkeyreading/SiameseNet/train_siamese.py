# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:56:34 2023

@author: Aakash
"""

import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import copy
from torch.optim import lr_scheduler
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import os
import pandas as pd
import clean_cornets  
#%%

class ImageDataset(Dataset):
	def __init__(self, img_folder, transform=None, stimcond = None):
		self.img_folder = img_folder
		self.transform = transform
		self.stimcond = stimcond

	def __len__(self):
		return len(self.stimcond)

	def __getitem__(self, idx):
		# Load the image
		img1_path = os.path.join(self.img_folder, self.stimcond.iloc[idx, 0])
		img2_path = os.path.join(self.img_folder, self.stimcond.iloc[idx, 1])
		target = int(self.stimcond.iloc[idx, 2])
		
		img1 = Image.open(img1_path).convert("RGB")
		img2 = Image.open(img2_path).convert("RGB")
	
		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
	
		return img1, img2, target

	
best_loss = 20
tr_loss = []; val_loss = []

dfs = []
allpaths = sorted(list(Path('conditionfiles').iterdir()))
for p in allpaths[:-1]:
	df = pd.read_excel(p)
	dfs.append(df) 
df_train = pd.concat(dfs, ignore_index = True)
df_val = pd.read_excel(allpaths[-1])

	
transform = {'train': transforms.Compose([
 			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


chosen_datasets = {}
chosen_datasets['train'] = ImageDataset('stimuli/train/', transform = transform['train'], stimcond = df_train)		 
chosen_datasets['val']   = ImageDataset('stimuli/val/',   transform = transform['train'], stimcond = df_val)
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = True) for x in ['train','val']}
dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val']}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
 
#%% Loading pretrained network trained on imagenet 
net = clean_cornets.CORnet_Z_tweak()
net.to(device)
checkpoint = torch.load('models/save_pre_z_49_full_nomir.pth.tar')['state_dict']
for key in list(checkpoint.keys()):
	if 'module.' in key:
		checkpoint[key.replace('module.', '')] = checkpoint[key]
		del checkpoint[key]
net.load_state_dict(checkpoint)
net = nn.Sequential(*list(net.children())[:-4])


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
num_epochs = 20
criterion = nn.BCELoss()
optimizer = optim.Adadelta(net.parameters(), lr=1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

tr_acc=[]
val_acc =[]
		
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	print('-' * 10)

	# Each epoch has a training and validation phase
	for phase in ['train', 'val']:
		if phase == 'train':
			net.train()  # Set model to training mode
		else:
			net.eval()   # Set model to evaluate mode

		running_loss = 0.0
		correct = 0

		# Iterate over data.
		for img1, img2, target in dataloaders[phase]:
			
			img1, img2, target = img1.to(device), img2.to(device), target.to(device)
			target = target.float()
			optimizer.zero_grad()

			# forward
			# track history if only in train
			with torch.set_grad_enabled(phase == 'train'):
				output1 = net(img1)
				output2 = net(img2)
				
				# Dissimilarity (1-r) after mean subtraction
				output1 = output1 - torch.mean(output1,1).expand(output1.size()[1],-1).T
				output2 = output2 - torch.mean(output2,1).expand(output2.size()[1],-1).T
				qdis = 1+1e-5-cos(output1, output2)

			# BCE loss
			outputs = 1-qdis/2 # To bring it into the range of 0 and 1
			loss = criterion(outputs, target)

			pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

			# backward + optimize only if in training phase
			if phase == 'train':
				loss.backward()
				optimizer.step()

			# statistics
			running_loss += loss.sum().item()*img1.size(0)
			print(loss.item())
			
		epoch_loss = running_loss / dataset_sizes[phase]
		if phase == 'train':
			exp_lr_scheduler.step()
			tr_loss.append(epoch_loss)
			tr_acc.append(correct/dataset_sizes[phase])
		else:
			val_loss.append(epoch_loss)
			val_acc.append(correct/dataset_sizes[phase])
   
		print('{} Loss: {:.4f}: {:2f}'.format(phase, epoch_loss, correct))
		# deep copy the model
		if phase == 'val':
			torch.save(net.state_dict(), 'models/cornet_lit_epoch'+str(epoch)+'.pth')

np.savez('models/training_loss.npz',tr_loss = tr_loss,val_loss = val_loss, tr_acc = tr_acc, val_acc=val_acc)



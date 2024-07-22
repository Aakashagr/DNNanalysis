#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:59:23 2022

@author: aakash
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets

#%%

data_dir = 'stimuli/AM/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
    
#%%
dataiter = iter(dataloaders['train'])

nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['v1'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)

stimtemp, classes = next(dataiter)
nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())

#%% plotting only the filter with maximum response

unitid = []
unitid.append(np.floor(np.argmax(nBli['v1'].detach().numpy(),1)/(56*56)))
unitid.append(np.floor(np.argmax(nBli['v2'].detach().numpy(),1)/(28*28)))
unitid.append(np.floor(np.argmax(nBli['v4'].detach().numpy(),1)/196))
unitid.append(np.floor(np.argmax(nBli['it'].detach().numpy(),1)/49))
unitid.append(['air','pain','square']) # stim category (assuming correct classification)

print(unitid)


#%% Plotting the data
positive = 1
fig, axes = plt.subplots(3, 5, figsize=(20, 10))

for i,name in enumerate(['V1','V2','V4','IT','output']):
	for j in range(3):
		if i == 4:
			image = plt.imread('plots/Activation_maximization_lucent/rep0/literate/AM_'+
					 name+'/'+unitid[i][j] +'.png')
		else:			
			image = plt.imread('plots/Activation_maximization_lucent/rep0/literate/AM_'+
						 name+'/'+str(int(unitid[i][j])).zfill(3) +'.png')
		
		if positive: 
			image = image[:,224:,:] # Positive direction
		else:
			image = image[:,:224,:] # Negative direction
			
		axes[j,i].imshow(image); 
		axes[j,i].axis('off')

fig.savefig('plots/temp_/negative.pdf')
plt.close(fig)
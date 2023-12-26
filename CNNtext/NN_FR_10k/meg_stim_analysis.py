# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:10:06 2023

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
from torchvision import datasets
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_ind

#%%

data_dir = 'stimuli/meg_stim/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 500,shuffle = False) for x in ['train']}
    
#%%
layers = ['v1','v2','v4','it','h','out']
metric = 'correlation'
R = np.zeros((6,5))
davg = np.zeros((6,2,5))

rep = 0

nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['out'] =[]
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('save/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
# checkpoint = torch.load('model_all-lang/save_lit_fr_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())

#%%
dmetric = 'correlation'

dcnn = np.zeros((114960,6)) 
for j,l in enumerate(layers):
	dcnn[:,j] = pdist(nBli[l].detach().numpy(),metric = dmetric)
	

#%%
plt.figure(figsize = (10,6))
plt.subplots_adjust(hspace=0.3, wspace = .3)


for i in range(6):
	ax = plt.subplot(2, 3, i + 1)

	ax.imshow(squareform(dcnn[:,i]))			 
	ax.set_title('Layer '+layers[i] + ' RDMs')

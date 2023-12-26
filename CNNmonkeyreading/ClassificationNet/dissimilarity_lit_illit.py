# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:10:43 2022

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
from torchvision import datasets
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr

#%%

data_dir = 'stimuli/neural_stim/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 500,shuffle = False) for x in ['train']}
    
#%%

nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['out'] =[]
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())


net = CORnet_Z_tweak()
checkpoint = torch.load('save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
ili = {};  ili['v1'] = [];  ili['v2'] = [];  ili['v4'] = []; ili['it'] = []; ili['h'] = []; ili['out'] = []
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
ili['v1'],  ili['v2'], ili['v4'], ili['it'], ili['h'], ili['out'] = net(stimtemp.float())
#%%
resp = nBli['v2'].detach().numpy()

dneu = np.zeros((36,8))
for w in range(36):
	for nw in range(8):
		r = pearsonr(resp[w*5+2], resp[180+w*8+nw])
		dneu[w,nw] = r[0]

plt.bar(range(8), np.mean(dneu,0))
plt.xticks(np.arange(8)-0.5, labels = ['all different letter',	'consonant/vowel string',	
							   'NW - diff. structure',	'W - diff. structure',	
							   'NW - same structure',	'W - same structure',	
							   'Transposition', 'Substitution'], rotation = 45)
plt.ylabel('Mean Similarity to word representation')
plt.title('Illiterate network')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:52:54 2022

@author: aakash
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle

layer = 'h'
#%%

with open('WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)

stimtype = ['PosTuning_letters','PosTuning_letters_lower']
meancoeff = np.zeros((len(wordSelUnit),26,8,2))

for qst, st in enumerate(stimtype):
    data_dir = 'stimuli/' + st
    transform = {'train': transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
    
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 300,shuffle = False) for x in ['train']}
    dataiter = iter(dataloaders['train'])
    
    nBli = {};
    net = CORNet_Z_nonbiased_words()
    checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    net.load_state_dict(checkpoint)    
    stimtemp, classes = next(dataiter)
    nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
        
    for i in range(len(wordSelUnit)):
        out = np.array(nBli[layer].detach().numpy())[:,wordSelUnit[i]]
        meancoeff[i,:,:,qst] = np.reshape(out,[26,8])

#%%
from scipy.stats import spearmanr
R = []
for i in range(np.shape(meancoeff)[0]):
    # r,p = spearmanr(meancoeff[i,:,:,0].flatten(), meancoeff[i,:,:,1].flatten())
    r,p = spearmanr(np.mean(meancoeff[i,:,:,0],1), np.mean(meancoeff[i,:,:,1],1))
    R.append(r)

#%%
medID = np.argsort(R)[len(R)//2]
# medID = np.argsort(R)

fig, axs = plt.subplots(1,2, figsize=(10,10), facecolor='w', edgecolor='k')
axs = axs.ravel();   
for i in range(2):
    charcoef = meancoeff[medID,:,:,i]
    maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
    for r in range(np.size(charcoef,0)):
        if i == 0:
            strchar = string.ascii_uppercase[r]
        else:
            strchar = string.ascii_lowercase[r]
        for c in range(np.size(charcoef,1)):
            strcol = 'red' if charcoef[r,c] >0 else 'blue'
            axs[i].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
            axs[i].set_xticks(np.arange(0.5,9,1)); axs[i].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
            axs[i].set_yticks(np.arange(0.5,27,1)); axs[i].set_yticklabels([]);
            axs[i].yaxis.set_ticks_position('none')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:13:51 2022

@author: aakash
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
from torchvision import datasets
import pickle
from scipy.stats import spearmanr, pearsonr
%config InlineBackend.figure_format = 'svg'
#%%

# data_dir = 'stimuli/CaseInvariance_rnd/'
data_dir = 'stimuli/CaseInvariance_letters/'

transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}


def activation(net, checkpoint,islit = 0):
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 500,shuffle = False) for x in ['train']}
    dataiter = iter(dataloaders['train'])
    
    nBli = {};
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    net.load_state_dict(checkpoint)    
    stimtemp, classes = next(dataiter)
    nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
            
    layer = ['v1','v2','v4','it','h','out']
    Ravg = []; Rstd = []
    if islit:
        wsname = 'WSunits/rep1/WSunits_lit_'
    else:
        wsname = 'WSunits/rep1/WSunits_ili_'
        
    for l in layer:        
        if l == 'out':
            out = np.array(nBli[l].detach().numpy())
        
        else:
            with open(wsname +l+'.pkl', 'rb') as f:
             	wordSelUnit = pickle.load(f)                            
            out = np.array(nBli[l].detach().numpy())[:,wordSelUnit]
                     
        R = []
        for i in range(np.shape(out)[1]):
            r,p = pearsonr(out[::2,i], out[1::2,i])
            R.append(r)
    
        Ravg.append(np.nanmean(R))
        Rstd.append(np.nanstd(R))
    
    return Ravg, Rstd
#%%

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
Ravg_lit, Rstd_lit = activation(net, checkpoint,1)

net = CORnet_Z_tweak()
checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
Ravg_ili, Rstd_ili = activation(net, checkpoint)

#%%
layer = ['V1','V2','V4','IT','H','OUT']


plt.figure(figsize=(10,8))
plt.bar(np.arange(6)-0.15, Ravg_lit, width = 0.3)
plt.bar(np.arange(6)+0.15, Ravg_ili, width = 0.3)

plt.errorbar(np.arange(6)-0.15, Ravg_lit, yerr=  Rstd_lit, fmt=".", color = 'k')
plt.errorbar(np.arange(6)+0.15, Ravg_ili, yerr=  Rstd_ili, fmt=".", color = 'k')
plt.xticks(ticks = range(6), labels = layer, size = 16)
plt.yticks(size = 16)
plt.ylim([0,1])
plt.xlabel('Layer', size = 18)
plt.ylabel('Correlation Coefficient', size = 18)
plt.legend(['Literate n/w', 'Illiterate n/w'], fontsize = 16)
plt.title('Case Invariant stimuli', size = 16)
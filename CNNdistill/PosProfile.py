#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:34:03 2022

@author: aakash
"""
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat
import os, sys
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
from clean_cornets import CORNet_Z_nonbiased_words

#%%
from torchvision import datasets, transforms

data_dir = 'stimuli/PosTuning/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(1):
    stimtemp, classes = next(dataiter)
    nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    # _,_,_,varIt,varh, varOut = net(stimtemp.float())
    # nBli['it'].extend(varIt.detach().numpy())
    # nBli['h'].extend(varh.detach().numpy())
    # nBli['out'].extend(varOut.detach().numpy())
    print(i)

#%%

wordlist = [i[3:] for i in chosen_datasets['train'].classes]
Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
Arel = [0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,0,2,4,6,1,3,5,7]

x = np.array(nBli['h'].detach().numpy())
nid = [282,270,23,53]
out = x[:,nid]

fig, axs = plt.subplots(2,len(nid), figsize=(20,10), facecolor='w', edgecolor='k')
axs = axs.ravel();
    
for i in range(len(nid)):
    qmat = np.zeros((8,4))
    for p in range(20):
        qmat[Arel[p], Rrel[p]] = out[p,i]
    axs[i].pcolor(qmat)
    axs[i].set_xticks(ticks = np.arange(4)+0.5); axs[i].set_xticklabels(['1','2','3','4']);
    axs[i].set_yticks(ticks = np.arange(8)+0.5); axs[i].set_yticklabels(['1','2','3','4','5','6','7','8']);
    axs[i].set_xlabel('Relative position')
    axs[i].set_ylabel('Absolute position')
    axs[i].set_title('Unit ID#: ' + str(nid[i]))

    qmat = np.zeros((8,4))
    for p in np.arange(20,28):
        qmat[Arel[p], Rrel[p]] = out[p,i]
    axs[i+4].pcolor(qmat)
    axs[i+4].set_xticks(ticks = np.arange(4)+0.5); axs[i+4].set_xticklabels(['1','2','3','4']);
    axs[i+4].set_yticks(ticks = np.arange(8)+0.5); axs[i+4].set_yticklabels(['1','2','3','4','5','6','7','8']);
    axs[i+4].set_xlabel('Relative position')
    axs[i+4].set_ylabel('Absolute position')



# plt.figure()
# plt.pcolor(out,cmap = 'jet')
# plt.yticks(ticks = np.arange(16)+.5, labels=wordlist);
# plt.xticks(ticks = np.arange(4)+.5, labels=['282','270','23','53'],rotation =45);

#%%

wordlist = [i[3:] for i in chosen_datasets['train'].classes]
Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
Arel = [0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,0,2,4,6,1,3,5,7]

x = np.array(nBli['it'].detach().numpy())
nid = [13850,13848,3022,7331]
out = x[:,nid]

fig, axs = plt.subplots(2,len(nid), figsize=(20,10), facecolor='w', edgecolor='k')
axs = axs.ravel();
    
for i in range(len(nid)):
    qmat = np.zeros((8,4))
    for p in range(20):
        qmat[Arel[p], Rrel[p]] = out[p,i]
    axs[i].pcolor(qmat)
    axs[i].set_xticks(ticks = np.arange(4)+0.5); axs[i].set_xticklabels(['1','2','3','4']);
    axs[i].set_yticks(ticks = np.arange(8)+0.5); axs[i].set_yticklabels(['1','2','3','4','5','6','7','8']);
    axs[i].set_xlabel('Relative position')
    axs[i].set_ylabel('Absolute position')
    axs[i].set_title('Unit ID#: ' + str(nid[i]))

    qmat = np.zeros((8,4))
    for p in np.arange(20,28):
        qmat[Arel[p], Rrel[p]] = out[p,i]
    axs[i+4].pcolor(qmat)
    axs[i+4].set_xticks(ticks = np.arange(4)+0.5); axs[i+4].set_xticklabels(['1','2','3','4']);
    axs[i+4].set_yticks(ticks = np.arange(8)+0.5); axs[i+4].set_yticklabels(['1','2','3','4','5','6','7','8']);
    axs[i+4].set_xlabel('Relative position')
    axs[i+4].set_ylabel('Absolute position')



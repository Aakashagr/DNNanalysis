#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:50:49 2022

@author: aakash
"""
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

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
from scipy import stats

# images = loadmat('baoimg.mat').get('img')
from clean_cornets import CORNet_Z_nonbiased_words,CORnet_Z_tweak

#%%
from torchvision import datasets, transforms

# data_dir = 'images/'
data_dir = 'stimuli/wordsets_1000cat_8ex/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}

wordlist = chosen_datasets['train'].classes

#%%
dataiter = iter(dataloaders['train'])

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
# net = CORnet_Z_tweak()
# checkpoint = torch.load('models/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
for i in range(25):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varIt,varh, varOut = net(stimtemp.float())
    nBli['v1'].extend(varv1.detach().numpy())
    nBli['v2'].extend(varv2.detach().numpy())
    nBli['v4'].extend(varv4.detach().numpy())
    nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['out'].extend(varOut.detach().numpy())
    print(i)

#%% Visual properties
wordlist = chosen_datasets['train'].classes
stimword = np.transpose(np.tile(wordlist,(8, 1))).flatten()
location = np.tile([1, 2, 3, 4, 1, 2, 3, 4],[1, len(wordlist)]).flatten()
size = np.tile([1, 1, 1, 1, 2, 2, 2, 2],[1, len(wordlist)]).flatten()
strlen = [len(i) for i in stimword] 

#%% Perfoming PCA

from sklearn.decomposition import PCA
npc = 10
pcax = PCA(n_components=npc)
out = pcax.fit_transform(nBli['v1'])

#%%

fig = plt.figure(figsize=(15,5))
titlab = ['location', 'size','string length']
map = [location, size, strlen]
npts = 2000
for i in range(3):
    ax = fig.add_subplot(1,3,i+1, projection='3d')
    ax.scatter3D(out[:npts, 0], out[:npts,1],out[:npts, 2], c = np.array(map[i][:npts]), cmap= 'rainbow',marker = 'o');
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # ax.set_xlim([-30,40]); ax.set_ylim([-20,30]); ax.set_zlim([-20,20]); 
    plt.title(titlab[i], fontsize = 15)
# fig.suptitle('V4 layer - Non-biased network', fontsize=18);
fig.suptitle('V1 layer - Illiterate network', fontsize=18);

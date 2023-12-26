# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:49:32 2022

@author: Aakash
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

data_dir = 'stimuli/wordsets_1000cat_8ex/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['out'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(40):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    _,_,_,varIt,varh, varOut = net(stimtemp.float())
    # nBli['it'].extend(varIt.detach().numpy())
    # nBli['h'].extend(varh.detach().numpy())
    nBli['out'].extend(varOut.detach().numpy())
    print(i)
	
	
#%%

pred = np.argmax(nBli['out'],1)

label = np.tile(np.arange(1000), [8,1]).T.flatten() + 1000
sum(pred == label)/8000

# pred_med = []

# for i in range(1000):
# 	pred_med.append(np.median(pred[i*8:(i+1)*8]))





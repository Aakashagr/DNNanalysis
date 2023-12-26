# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:15:52 2023

@author: Aakash
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
from sklearn.linear_model import LassoCV, RidgeCV
from scipy import stats

from clean_cornets import CORNet_Z_nonbiased_words

#%%
from torchvision import datasets, transforms

transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

data_dir = 'words/'
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform['train']) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}
wordlist = chosen_datasets['train'].classes


data_dir = 'words_malabi/'
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform['train']) for x in ['train']}
dataloaders_m = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}
wordlist_m = chosen_datasets['train'].classes

#%%

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

dataiter = iter(dataloaders['train'])
lit_tr = {}; lit_tr['h'] = []; 
for i in range(100):
    stimtemp, classes = next(dataiter)
    _,_,_,_,varh, _ = net(stimtemp.float())
    lit_tr['h'].extend(varh.detach().numpy())
    print(i)
	

dataiter = iter(dataloaders_m['train'])
lit_ts = {}; lit_ts['h'] = []; 
for i in range(2):
    stimtemp, classes = next(dataiter)
    _,_,_,_,varh, _ = net(stimtemp.float())
    lit_ts['h'].extend(varh.detach().numpy())
    print(i)
#%%

pcax = PCA(n_components=40)

tr_pattern = np.array(lit_tr['h'])
pcax.fit(tr_pattern)
pc_tr = pcax.transform(tr_pattern)
pc_tr = np.c_[pc_tr, np.ones((np.shape(pc_tr)[0],1))]
tr_label = np.identity(10000)
weights = (tr_label@np.linalg.pinv(pc_tr).T).T
  

ts_pattern = np.array(lit_ts['h'])
pc_ts = pcax.transform(ts_pattern)
pc_ts = np.c_[pc_ts , np.ones((np.shape(pc_ts)[0],1))]
test_pred = pc_ts@weights
		
np.array(wordlist)[np.argmax(test_pred,1)]
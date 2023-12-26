# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:17:18 2021

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
from training_code import clean_cornets

#%%
from torchvision import datasets, transforms

data_dir = 'stimuli/HierStim/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}

qo = np.array(np.arange(0,400));
qe = np.array(np.arange(400,1660));
qf = np.array(np.arange(1660,2920));

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
nBli['it'] = []; nBli['h'] = []; nBli['out'] = []

net = clean_cornets.CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
net.eval()

for i in range(37):
    stimtemp, classes = next(dataiter)
    varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
    nBli['v1'].extend(varV1.detach().numpy())
    nBli['v2'].extend(varV2.detach().numpy())
    nBli['v4'].extend(varV4.detach().numpy())
    nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['out'].extend(varOut.detach().numpy())
    print(i)


#%%
nid = {}
nid[0] = []
nid[1] = []
nid[2] =  [6,   9,  11,  12,  15,  17,  27,  28,  29,  39,  40,  55,  65,
        73,  75,  89, 123, 124, 150, 155, 169, 181, 189, 197, 201, 220,
       226, 241, 242, 243, 249, 250, 258, 282, 283, 299, 300, 304, 325,
       341, 351, 354, 355, 361, 363, 367, 375, 381, 385, 395, 414, 422,
       423, 427, 430, 437, 442, 482, 485, 488, 496, 498, 506]
type = ['English','French','English+French']
data_d = np.array(nBli['h'])

#%% # Plotting the mean response averaged across all units
color = loadmat('7Tbilingual_colorcode.mat')['bilingualcolormap'][:,1:]
color = np.concatenate(([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],color))

label = ['bodies','faces','houses','tools','L_lf_le','L_lf_he','L_hf_le','L_hf_he',
         'B_lf_le','B_lf_he','B_hf_le','B_hf_he','Q_lf_le','Q_lf_he','Q_hf_le','Q_hf_he',
          'E_words', 'F_words']

fig, axs = plt.subplots(1,3, figsize=(30,5), facecolor='w', edgecolor='k')
axs = axs.ravel();


for i in range(3):
	if len(nid[i]):
	    meanresp = []
	    for t in range(4):
	        meanresp.append(np.nanmean(np.nanmean(data_d[np.arange(t*100,(t+1)*100), np.reshape(nid[i],[np.size(nid[i]),1])  ],1),0))
	    for t in range(14):
	        meanresp.append(np.nanmean(np.nanmean(data_d[np.arange(400+t*180,400+(t+1)*180),np.reshape(nid[i],[np.size(nid[i]),1])],1),0))

	    axs[i].bar(range(18),meanresp,color = color)
	    axs[i].set_xticks(range(18)); axs[i].set_xticklabels(label,rotation = 45)
	    axs[i].set_ylabel('Mean response')
	    axs[i].set_title(type[i]+ ' word seletive units - H layer')
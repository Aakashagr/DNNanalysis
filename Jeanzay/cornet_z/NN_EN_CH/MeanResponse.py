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
qc = np.array(np.arange(1660,2920));
#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
nBli['it'] = []; nBli['h'] = []; nBli['out'] = []

net = clean_cornets.CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location=torch.device('cpu'))['state_dict']

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
# Using predefined word selective units
neuid_e = [ 12,  17,  56,  65, 131, 193, 277, 376, 422, 430, 488]

neuid_c = [ 37,  58,  90,  95, 112, 145, 165, 186, 206, 221, 229, 247, 249,
       250, 259, 275, 291, 298, 331, 351, 386, 391, 416, 417, 423, 449,
       458, 463, 502, 503]

neuid_ec = [  5,   6,  11,  12,  15,  17,  27,  29,  37,  39,  40,  41,  47,
        49,  55,  56,  58,  65,  71,  73,  75,  78,  89,  90,  91,  95,
       111, 112, 123, 124, 128, 131, 137, 145, 150, 155, 163, 165, 169,
       173, 179, 180, 186, 190, 193, 197, 201, 203, 206, 218, 220, 221,
       222, 229, 231, 241, 243, 245, 247, 248, 249, 250, 259, 273, 275,
       277, 283, 291, 298, 300, 306, 313, 331, 337, 341, 343, 351, 352,
       355, 356, 357, 361, 363, 365, 375, 376, 378, 381, 382, 385, 386,
       391, 392, 414, 415, 416, 417, 422, 423, 427, 428, 430, 435, 437,
       449, 458, 461, 463, 473, 476, 481, 485, 488, 496, 497, 498, 502,
       503, 506]

nid = {}
nid[0] = neuid_e
nid[1] = neuid_c
nid[2]= list(set(neuid_ec)^set(neuid_c)^set(neuid_e))
type = ['English','Chinese','English+Chinese']

data_d = np.array(nBli['h'])

#%% Plotting the mean response averaged across all units
color = loadmat('7TbilingualCE_colorcode.mat')['bilingualCEcolormap'][:,1:]
color = np.concatenate(([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],color))

label = ['bodies','faces','houses','tools','L-','L+','B-','B+','Q-','Q+',
          'E_words', 'stroke','Nrad','Nchar1','Nchar2','NwReal','LF','HF']

fig, axs = plt.subplots(1,3, figsize=(30,5), facecolor='w', edgecolor='k')
axs = axs.ravel();

for i in range(3):
    meanresp = []
    for t in range(4):
        meanresp.append(np.nanmean(np.nanmean(data_d[np.arange(t*100,(t+1)*100), np.reshape(nid[i],[np.size(nid[i]),1])  ],1),0))
    for t in range(14):
        meanresp.append(np.nanmean(np.nanmean(data_d[np.arange(400+t*180,400+(t+1)*180),np.reshape(nid[i],[np.size(nid[i]),1])],1),0))


    axs[i].bar(range(18),meanresp,color = color)
    axs[i].set_xticks(range(18)); axs[i].set_xticklabels(label,rotation = 45)
    axs[i].set_ylabel('Mean response')
    axs[i].set_title(type[i]+ ' word seletive units - h layer')

#%% Plotting the mean response for each unit in h layer
if 0:
	fig, axs = plt.subplots(8,5, figsize=(30,40), facecolor='w', edgecolor='k')
	axs = axs.ravel();

	color = ['k','k','k','k','b','c','green','limegreen','orange','gold','tomato','steelblue','skyblue','slateblue','orchid','lightcoral','r','maroon']
	for n,nid in enumerate(np.sort(list(set(neuid_c)^set(neuid_ec)))):
	# for n,nid in enumerate(np.sort(neuid_fc)):
	    meanresp = []
	    for t in range(4):
	        meanresp.append(np.nanmean(data_d[range(t*100,(t+1)*100),nid],0))
	    for t in range(14):
	        meanresp.append(np.nanmean(data_d[range(400+t*180,400+(t+1)*180),nid],0))

	    axs[n].bar(range(18),meanresp,color = color)
	    axs[n].set_title('unit id: ' + str(nid),fontsize = 24)
	fig.suptitle('Chinese selective units', fontsize = 32)

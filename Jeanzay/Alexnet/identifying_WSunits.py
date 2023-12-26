# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:52:33 2021

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
import sys, os
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
from torchvision import datasets, transforms
from alexnet_def_feat import AlexNet

data_dir = 'stimuli/wordselective_stimuli/'
transform = {'train': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])

#%%

net = AlexNet(num_classes = 2000)
checkpoint = torch.load('save/save_Alexnet_lit_29.pth.tar')['state_dict']
# checkpoint = torch.load('alexnet_trained.pth')
model_dict =["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight",
			 "conv3.0.bias", "conv4.0.weight", "conv4.0.bias", "conv5.0.weight", "conv5.0.bias",
			 "fc6.1.weight", "fc6.1.bias", "fc7.1.weight", "fc7.1.bias", "fc8.1.weight", "fc8.1.bias"]

state_dict={}; i=0
for k,v in checkpoint.items():
	state_dict[model_dict[i]] =  v
	i+=1

net.load_state_dict(state_dict); net.eval()
for param in net.parameters():
    param.requires_grad = False

feat = {}; feat[1] = []; feat[2] = []; feat[3] = []

for i in range(10):
    stimtemp, classes = next(dataiter)
    _,_,_,_,_,L6,L7,L8 = net(stimtemp.float())
    feat[1].extend(L6.detach().numpy())
    feat[2].extend(L7.detach().numpy())
    feat[3].extend(L8.detach().numpy())
    print(i)

#%%
# Identify word selective units based on 3 standard deviations above the mean
qo = np.array(np.arange(0,400));
qf = np.array(np.arange(400,800));
data_d = np.array(feat[3])

neuid_f = [];
for unit in range(np.size(data_d,1)):
    Frmean = np.mean(data_d[qf,unit])
    Objmean   = np.mean(data_d[qo,unit])
    Objstdev  = np.var(data_d[qo,unit])**0.5

    if Frmean >= Objmean + 3*Objstdev:
        neuid_f.append(unit)

print(['% of French word-selective units: '+ str(np.size(neuid_f)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_f))])


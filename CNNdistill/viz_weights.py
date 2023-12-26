#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 10:19:19 2022

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

#%%

net = CORnet_Z_tweak()
checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
# net = CORNet_Z_nonbiased_words()
# checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)

#%%

Wmat = net.V1.state_dict()['conv.weight'].numpy()
Wmat = np.transpose(Wmat,[0,2,3,1])

#%%
fig, axs = plt.subplots(8,8, figsize=(20,20), facecolor='w', edgecolor='k')
axs = axs.ravel();    

for i in range(64):
    qt = Wmat[i]
    qt_max = max(qt.flatten())
    qt_min = min(qt.flatten())
    qt = (qt - qt_min)/(qt_max - qt_min)
    axs[i].imshow(qt)
    axs[i].axis('off')

#%% Comparing change in weights between literate and illiterate network

net_ili = CORnet_Z_tweak()
checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net_ili.load_state_dict(checkpoint)

net_lit = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net_lit.load_state_dict(checkpoint)


from scipy.stats import spearmanr 
R = np.zeros((4,1))
R[0],_ = spearmanr(net_lit.V1.state_dict()['conv.weight'].numpy().flatten() , net_ili.V1.state_dict()['conv.weight'].numpy().flatten())
R[1],_ = spearmanr(net_lit.V2.state_dict()['conv.weight'].numpy().flatten() , net_ili.V2.state_dict()['conv.weight'].numpy().flatten())
R[2],_ = spearmanr(net_lit.V4.state_dict()['conv.weight'].numpy().flatten() , net_ili.V4.state_dict()['conv.weight'].numpy().flatten())
R[3],_ = spearmanr(net_lit.IT.state_dict()['conv.weight'].numpy().flatten() , net_ili.IT.state_dict()['conv.weight'].numpy().flatten())


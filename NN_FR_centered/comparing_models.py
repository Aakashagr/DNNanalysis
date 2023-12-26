#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:57:13 2022

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
from sklearn.linear_model import LassoCV, RidgeCV
from scipy import stats

from training_code.clean_cornets import CORNet_Z_nonbiased_words

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
checkpoint = torch.load('save_rep2/save_lit_no_bias_z_54_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

nBli = {}; nBli['v1'] = []; nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; 
for i in range(100):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    _,_,_,_,varh, _ = net(stimtemp.float())
    # nBli['v1'].extend(varv4.detach().numpy())
    # nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    print(i)
#%%
import pickle
with open('WSunits_lit.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)
     
# Comparing both ends models
stimword = np.transpose(np.tile(wordlist,(8, 1))).flatten()
x = np.array(nBli['h']) #stimulus 


# Setting up the parameters to perform PC
out = x[:,wordSelUnit]  # Analysing properties of individual word selective units
strlist = list(string.ascii_lowercase) # storing charaacters from A-Z

# Building the regression matrix for non-overlapping ends
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
    mpt = round(len(seq)/2)
    for j, char in enumerate(seq[:mpt]):  # Left half of the words 
        if char in strlist:
            pid = (strlist.index(char)*8) + j
            Xmat[i,pid] += 1 
            
    for j, char in enumerate(seq[mpt:][::-1]):  # right half of the words 
        if char in strlist:
            pid = (strlist.index(char)*8) + 7 - j
            Xmat[i,pid] += 1 

rfit_nol = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
    yobs = out[:,npc]       
    reg = LassoCV(cv=5, random_state=0).fit(Xmat, yobs)   
    corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
    rfit_nol[npc] = corrval**2
    
print('Non-overlapping both-ends model done')  


###########  Overlapping end coding model
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
    p1 = np.min([4,len(seq)])
    for j, char in enumerate(seq[:p1]):  # Left part 
        if char in strlist:
            pid = (strlist.index(char)*8) + j
            Xmat[i,pid] += 1 
            
    for j, char in enumerate(seq[-p1:][::-1]):  # Right part 
        if char in strlist:
            pid = (strlist.index(char)*8) + 7 - j
            Xmat[i,pid] += 1

rfit_ol = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
    yobs = out[:,npc]       
    reg = LassoCV(cv=5, random_state=0).fit(Xmat, yobs)   
    corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
    rfit_ol[npc] = corrval**2
  
plt.figure()
plt.scatter(rfit_nol, rfit_ol); plt.plot([0,1],[0,1])
plt.xlabel('Non-overlapping end model')
plt.ylabel('Overlapping end model')

#%%
# Fitting retinotopic models
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
    offset = int(4-len(seq)/2)
    for j, char in enumerate(seq[:mpt]):  # Left half of the words 
        if char in strlist:
            pid = (strlist.index(char)*8) + j + offset
            Xmat[i,pid] += 1 

rfit_ret = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
    yobs = out[:,npc]       
    reg = LassoCV(cv=5, random_state=0).fit(Xmat, yobs)   
    corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
    rfit_ret[npc] = corrval**2
 
plt.figure()
plt.scatter(rfit_nol, rfit_ret); plt.plot([0,1],[0,1])
plt.xlabel('Non-overlapping end model')
plt.ylabel('Word Centred model')

#%%# Comparing left-right vs right-left model fits

# Building the regression matrix
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
    for j, char in enumerate(seq): 
        if char in strlist:
            pid = (strlist.index(char)*8) + j
            Xmat[i,pid] += 1 

rfit_lr = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
    yobs = out[:,npc]       
    reg = LassoCV(cv=5, random_state=0).fit(Xmat, yobs)   
    corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
    rfit_lr[npc] = corrval**2
    
print('LR model done')  
###########  Filling the location from right to left
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
    pos_offset = 8 - len(seq)
    for j, char in enumerate(seq):
        if char in strlist:
            pid = (strlist.index(char)*8) + j + pos_offset
            Xmat[i,pid] += 1 


rfit_rl = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
    yobs = out[:,npc]       
    reg = LassoCV(cv=5, random_state=0).fit(Xmat, yobs)   
    corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
    rfit_rl[npc] = corrval**2
 
plt.figure()
plt.scatter(rfit_lr, rfit_rl); plt.plot([0,1],[0,1])
plt.xlabel('Left to Right model')
plt.ylabel('Right to Left model')

plt.figure()
plt.scatter(rfit_lr, rfit_nol); plt.plot([0,1],[0,1])
plt.xlabel('Left to Right model')
plt.ylabel('Non-overlapping end model')

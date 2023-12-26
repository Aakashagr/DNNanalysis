#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:50:26 2022

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

data_dir = 'stimuli/bigrams/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
    
    #%%

nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['out'] =[]
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())



net = CORnet_Z_tweak()
checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
ili = {};  ili['v1'] = [];  ili['v2'] = [];  ili['v4'] = []; ili['it'] = []; ili['h'] = []; ili['out'] = []
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
ili['v1'],  ili['v2'], ili['v4'], ili['it'], ili['h'], ili['out'] = net(stimtemp.float())



#%%
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, ttest_rel
layers = ['v1','v2','v4','it','h','out']
metric = 'correlation'
R = np.zeros((6,1))
davg = np.zeros((6,2))
pval = np.zeros((6,1))
for i,l in enumerate(layers):
    dili = pdist(ili[l].detach().numpy(),metric = metric)
    dlit = pdist(nBli[l].detach().numpy(), metric = metric)
    R[i],_ = spearmanr(dili, dlit)    
    davg[i,:] = [np.mean(dili), np.mean(dlit)]
    pval[i] = ttest_rel(dlit, dili)[1]

plt.bar(range(6),R.flatten())
plt.xticks(range(6),labels = layers, size = 16)
plt.ylabel('Correlation coefficient', size = 16)
plt.title('Match between literate and illiterate networks')
   

plt.figure()
plt.bar(np.arange(6)-.2,davg[:,0], width =0.4)
plt.bar(np.arange(6)+.2,davg[:,1], width =0.4)
plt.xticks(range(6),labels = layers, size = 16)
plt.ylabel('Mean dissimilarity')
plt.legend(['Illiterate n/w','Literate n/w'])
#%% Building compositional part-sum model

import itertools
from sklearn.linear_model import LinearRegression
sid = [p for p in itertools.product(range(7), repeat=2)] # stimuli index
lid = np.array(list(itertools.combinations(range(7), 2))) # Letter pair index
bid = list(itertools.combinations(range(49), 2)) # bigram pair index

Xmat = np.zeros((len(bid),len(lid)*4))   
for i in range(np.shape(Xmat)[0]):
    s1 = sid[bid[i][0]]
    s2 = sid[bid[i][1]]
    
    c1 = np.where(np.all(lid == np.sort([s1[0],s2[0]]),axis=1))[0]
    c2 = np.where(np.all(lid == np.sort([s1[1],s2[1]]),axis=1))[0]
    a1 = np.where(np.all(lid == np.sort([s1[0],s2[1]]),axis=1))[0]
    a2 = np.where(np.all(lid == np.sort([s1[1],s2[0]]),axis=1))[0]
    w1 = np.where(np.all(lid == np.sort([s1[0],s1[1]]),axis=1))[0]
    w2 = np.where(np.all(lid == np.sort([s2[0],s2[1]]),axis=1))[0]
    
    
    if len(c1): 
        Xmat[i,c1[0] + len(lid)*0] += 1
    if len(c2): 
        Xmat[i,c2[0] + len(lid)*1] += 1     
    if len(a1): 
        Xmat[i,a1[0] + len(lid)*2] += 1
    if len(a2): 
        Xmat[i,a2[0] + len(lid)*2] += 1  
    if len(w1): 
        Xmat[i,w1[0] + len(lid)*3] += 1
    if len(w2): 
        Xmat[i,w2[0] + len(lid)*3] += 1  
    
#%%  
metric = 'correlation'
r2 = np.zeros((6,2))
coeff = np.zeros((6,len(lid)*4+1, 2))
for i,l in enumerate(layers):
    dili = pdist(ili[l].detach().numpy(),metric = metric)
    dlit = pdist(nBli[l].detach().numpy(), metric = metric)

    reg = LinearRegression().fit(Xmat, dili)
    r2[i,0] = reg.score(Xmat, dili)
    coeff[i,:,0] = np.r_[reg.coef_, reg.intercept_]
        
    reg = LinearRegression().fit(Xmat, dlit)
    r2[i,1] = reg.score(Xmat, dlit)
    coeff[i,:,1] = np.r_[reg.coef_, reg.intercept_]
    
    
plt.figure()
plt.bar(np.arange(6)-.2,r2[:,0], width =0.4)
plt.bar(np.arange(6)+.2,r2[:,1], width =0.4)
plt.xticks(range(6),labels = layers, size = 16)
plt.ylabel('Model fit R2')
plt.legend(['Illiterate n/w','Literate n/w'], loc = 'lower left')

# Plotting the coeff
for i,l in enumerate(layers):
    clit = np.mean(np.reshape(coeff[i,:-1,1],[4,21]),1)
    cili = np.mean(np.reshape(coeff[i,:-1,0],[4,21]),1)
    
    plt.figure()
    plt.bar(np.arange(4)-.2,cili, width =0.4)
    plt.bar(np.arange(4)+.2,clit, width =0.4)
    plt.xticks(range(4),labels = ['C1','C2','A','W'], size = 16)
    plt.ylabel('Model coeff')
    plt.legend(['Illiterate n/w','Literate n/w'])
    plt.title(l)
 

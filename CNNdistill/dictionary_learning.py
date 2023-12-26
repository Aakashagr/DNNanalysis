#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:24:02 2022

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
from torchvision import datasets, transforms
import pickle
#%%
layer = 'v2'

with open('WSunits/WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)

os.makedirs('plots/postuning_'+layer+'/', exist_ok=True)

qmat = np.zeros((len(wordSelUnit),5,4))
cnt = 0
for i, nid in enumerate(wordSelUnit):
    folderid = 'WSunit_' + str(i).zfill(3)
    print(folderid)

    data_dir = 'stimuli/PosTuning_allunits_'+layer+'/'
    transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
    
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
    dataiter = iter(dataloaders[folderid])
    
    nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['v1'] = []
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
    
    
    wordlist = [i[3:] for i in chosen_datasets[folderid].classes]
    Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    Arel = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,1,1,1,1]
    Rrel = np.tile(Rrel,2).flatten(); 
    Arel = np.tile(Arel,2).flatten(); 

    x = nBli[layer].detach().numpy()
    out = x[:,nid]

    for p in np.arange(0,20):
        qmat[cnt][Arel[p], Rrel[p]] = out[p]
    cnt += 1 
#%%
from sklearn.decomposition import NMF, DictionaryLearning
X = np.reshape(qmat,[np.shape(qmat)[0],20])

# model = NMF(n_components=20, init='random', random_state=0, alpha= 0.5)
model = DictionaryLearning(
    n_components=15, transform_algorithm='lasso_lars', transform_alpha=0.1,fit_algorithm = 'cd',
    random_state=42, n_jobs = -1,positive_code = True, positive_dict = True
    )
W = model.fit_transform(X)
H = model.components_

H_new = np.reshape(H,[np.shape(H)[0],5,4])
# print(model.reconstruction_err_)
#%%
for i in range(np.shape(H)[0]):
    plt.figure()
    plt.pcolor(H_new[i])
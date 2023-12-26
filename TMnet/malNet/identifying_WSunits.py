# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:24:08 2021

Identifying word selective units
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
from torchvision import datasets, transforms
import clean_cornets

#%%
data_dir = 'stimuli/wordselective_stimuli/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['v1'] = []; nBli['v2'] = [];  nBli['v4'] = [];   nBli['it'] = [];  nBli['h'] = [];

net = clean_cornets.CORNet_Z_nonbiased_words()
checkpoint = torch.load('save/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
# net = clean_cornets.CORnet_Z_tweak()
# checkpoint = torch.load('models/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']


for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(10):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1, varv2, varv4, varit, varh, _ = net(stimtemp.float())
    nBli['v1'].extend(varv1.detach().numpy())
    nBli['v2'].extend(varv2.detach().numpy())
    nBli['v4'].extend(varv4.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['it'].extend(varit.detach().numpy())
    print(i)

#%% Identify word selective units based on 3 standard deviations above the mean
qo = np.array(np.arange(0,400));
qf = np.array(np.arange(400,800));

layer = ['v1','v2','v4','it','h']
import pickle

for l in layer:
    data_d = np.array(nBli[l])
    
    neuid = []; 
    for unit in range(np.size(data_d,1)):
        Frmean = np.mean(data_d[qf,unit])
        Objmean   = np.mean(data_d[qo,unit])
        Objstdev  = np.var(data_d[qo,unit])**0.5
    
        if np.var(data_d[qf,unit]) > 1:
            if Frmean > Objmean + 3*Objstdev:
                neuid.append(unit)
    
    print(['% of Malayalam word-selective units in '+l+' :'+ str(np.size(neuid)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid))])
    
    with open('WSunits/WSunits_lit_'+l+'.pkl', 'wb') as f:
     	pickle.dump(neuid, f)
	
#%% 

scores = np.load('save/cat_scores_lit_no_bias_z_full_nomir.npy')
plt.plot(np.mean(scores[:,:1000],1))
plt.plot(np.mean(scores[:,1000:],1))
plt.xlabel('epochs')
plt.ylabel('Accuracy')
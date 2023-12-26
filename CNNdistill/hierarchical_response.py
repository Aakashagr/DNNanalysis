#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:59:23 2022

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

data_dir = 'stimuli/vinckier/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
    
    #%%
dataiter = iter(dataloaders['train'])

nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['v1'] = []
net = CORnet_Z_tweak()
checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
# net = CORNet_Z_nonbiased_words()
# checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(14):
    print(i)
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
    nBli['v1'].extend(varv1.detach().numpy())
    nBli['v2'].extend(varv2.detach().numpy())
    nBli['v4'].extend(varv4.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['it'].extend(varit.detach().numpy())
    
#%%
layers = ['v1','v2','v4','it','h']

for layer in layers:
    with open('WSunits/rep1/WSunits_ili_'+layer+'.pkl', 'rb') as f:
     	wordSelUnit = pickle.load(f)
         
    out = np.array(nBli[layer])[:,wordSelUnit]
    RavgN = []
    Ravg = []
    Rsem = []
    
    
    for i in range(4):
        qt = np.mean(out[i*100:(i+1)*100,:],1)
        RavgN.append(qt)
        
    
    for i in range(6):
        qt = np.mean(out[i*400 + 400: i*400 + 500,:],1)
        RavgN.append(qt)        
    
    
    for i in range(4):
        qt = np.mean(out[i*100:(i+1)*100,:],0)/np.max(RavgN)
        Ravg.append(np.mean(qt))
        Rsem.append(np.std(qt)/10)
        
    
    for i in range(6):
        qt = np.mean(out[i*400 + 400: i*400 + 500,:],0)/np.max(RavgN)
        Ravg.append(np.mean(qt))
        Rsem.append(np.std(qt)/10)
        
    readable = ['faces', 'bodies', 'houses', 'tools', 'fonts', 'inf.let.', 'fre.let.', 'bigrams', 'quad.', 'words']
    colors = ['red', 'violet', 'yellow', 'blue', 'aquamarine', 'mediumaquamarine', 'lightgreen', 'lime', 'forestgreen',
                  'darkgreen']
        
    plt.figure(figsize = (4,5))
    plt.bar(range(10),Ravg, color = colors)
    plt.xticks(range(10), labels = readable, rotation =45)
    plt.errorbar(range(10), Ravg, yerr = Rsem,fmt = '.', color = 'k')
    plt.title(layer.upper()+ ', #units: ' + str(len(wordSelUnit)) + ', %units = ' + '{:.2f}'.format(len(wordSelUnit)*100/np.shape(np.array(nBli[layer]))[1]))
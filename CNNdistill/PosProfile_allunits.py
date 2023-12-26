#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:54:38 2022

@author: aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle
#%%
layer = 'v4'

with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)

os.makedirs('plots/postuning_'+layer+'/', exist_ok=True)


for i, nid in enumerate(wordSelUnit):
    folderid = 'WSunit_' + str(i).zfill(3)
    print(folderid)

    data_dir = 'stimuli/PosTuning_allunits_'+layer+'/'
    transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
    
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
    
    #%%
    dataiter = iter(dataloaders[folderid])
    
    nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['v4'] = []
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
        # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
        varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
        # nBli['v1'].extend(varv1.detach().numpy())
        # nBli['it'].extend(varit.detach().numpy())
        nBli[layer].extend(varv4.detach().numpy())
    
    #%%
    
    wordlist = [i[3:] for i in chosen_datasets[folderid].classes]
    Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,0,1,2,0,1,2,0,1,2]
    Arel = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,1,1,1,2,2,2,3,3,3]
    Rrel = np.tile(Rrel,2).flatten(); 
    Arel = np.tile(Arel,2).flatten(); 

    x = np.array(nBli[layer])
    out = x[:,nid]
    
    if np.max(out) < 4:
        continue
    
    fig, axs = plt.subplots(2,2, figsize=(7,9), facecolor='w', edgecolor='k')
    axs = axs.ravel();
    sid = {}; 
    sid[0] = np.arange(0,20); sid[1] = np.arange(20,32)
    sid[2] = np.arange(32,52); sid[3] = np.arange(52,64)

    for i in range(4):
        qmat = np.zeros((5,4))
        for p in sid[i]:
            qmat[Arel[p], Rrel[p]] = out[p]
        im = axs[i].pcolor(qmat, cmap = 'Blues')
        axs[i].set_xticks(ticks = np.arange(4)+0.5); axs[i].set_xticklabels(['1','2','3','4']);
        axs[i].set_yticks(ticks = np.arange(5)+0.5); axs[i].set_yticklabels(['1','2','3','4','5']);
        axs[i].set_xlabel('Relative letter position')
        axs[i].set_ylabel('Absolute word position')
        axs[i].set_title('Unit ID#: ' + str(nid) + ', stim: ' + wordlist[sid[i][0]])
        im.set_clim(0,max(out)+2) 
       
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)    
    
    fig.savefig('plots/postuning_'+layer+'/' +folderid + '.pdf')
    fig.savefig('plots/postuning_'+layer+'/' +folderid + '.png')
    plt.close(fig)

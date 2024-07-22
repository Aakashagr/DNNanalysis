#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:04:33 2022

@author: aakash
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle

layer = 'v4'
#%%

with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)


data_dir = 'stimuli/PosTuning_letters/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
    
    #%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['v4'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(2):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
    nBli['v4'].extend(varv4.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['it'].extend(varit.detach().numpy())
    
#%%
os.makedirs('plots/letter_tuning_'+layer+'_rf/', exist_ok= True)
meancoeff = np.zeros((len(wordSelUnit),26))

for bias in np.arange(0,len(wordSelUnit),10):

    fig, axs = plt.subplots(1,10, figsize=(40,10), facecolor='w', edgecolor='k')
    axs = axs.ravel();    
    for i in np.arange(bias,10+bias):

        # Visualizing the coefficients
        out = np.array(nBli[layer])[:,wordSelUnit[i]]
        charcoef = np.reshape(out,[26,8])
        meancoeff[i,:] = np.max(charcoef,1)
        maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
        for r in range(np.size(charcoef,0)):
    #         strchar = string.ascii_lowercase[r]
            strchar = string.ascii_uppercase[r]
            for c in range(np.size(charcoef,1)):
                strcol = 'red' if charcoef[r,c] >0 else 'blue'
                axs[i-bias].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
                axs[i-bias].set_xticks(np.arange(0.5,9,1)); axs[i-bias].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
                axs[i-bias].set_yticks(np.arange(0.5,27,1)); axs[i-bias].set_yticklabels([]);
                axs[i-bias].yaxis.set_ticks_position('none')

    
        axs[i-bias].set_title('unit #: ' + str(wordSelUnit[i]), fontsize = 16)
        
    fig.savefig('plots/letter_tuning_'+layer+'_rf/' +str(bias) + '.png')
    plt.close(fig)
        

    
import pickle
with open('mean_response_coeff/mean_lresponse_'+ layer+'.pkl', 'wb') as f:
 	pickle.dump(meancoeff, f)
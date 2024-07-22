# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:17:52 2024

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
from torchvision import datasets
import pickle
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, ttest_ind

#%%

data_dir = 'stimuli/PosTuning_letters/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 300,shuffle = False) for x in ['train']}
    
#%%
layers = ['v1','v2','v4','it','h','out']
metric = 'correlation'
R = np.zeros((6,5))
davg = np.zeros((6,2,5))

for rep in range(5):
	nBli = {};  nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; nBli['out'] =[]
	net = CORNet_Z_nonbiased_words()
	checkpoint = torch.load('model_all-lang/save_lit_fr_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)
	dataiter = iter(dataloaders['train'])
	stimtemp, classes = next(dataiter)
	nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())



	net = CORnet_Z_tweak()
	checkpoint = torch.load('model_all-lang/save_lit_illit_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
	ili = {};  ili['v1'] = [];  ili['v2'] = [];  ili['v4'] = []; ili['it'] = []; ili['h'] = []; ili['out'] = []
	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)
	dataiter = iter(dataloaders['train'])
	stimtemp, classes = next(dataiter)
	ili['v1'],  ili['v2'], ili['v4'], ili['it'], ili['h'], ili['out'] = net(stimtemp.float())


	for i,l in enumerate(layers):
	    dili = pdist(ili[l].detach().numpy()[3::8,:],metric = metric)
	    dlit = pdist(nBli[l].detach().numpy()[3::8,:], metric = metric)
	    R[i,rep],_ = spearmanr(dili, dlit)    
	    davg[i,:,rep] = [np.mean(dili), np.mean(dlit)]

#%%
plt.bar(range(6),np.mean(R,1),yerr = np.std(R,1))
plt.xticks(range(6),labels = layers, size = 16)
plt.ylabel('Correlation coefficient', size = 16)
plt.title('Match between literate and illiterate networks')
   
	
plt.figure()
plt.bar(np.arange(6)-.2,np.mean(davg[:,0,:],1), yerr = np.std(davg[:,0,:],1), width =0.4)
plt.bar(np.arange(6)+.2,np.mean(davg[:,1,:],1), yerr = np.std(davg[:,1,:],1), width =0.4)
plt.ylim([0,0.6])
plt.xticks(range(6),labels = layers, size = 16)
plt.ylabel('Mean dissimilarity')
plt.legend(['Illiterate n/w','Literate n/w'])
plt.savefig('plots/Dissimilarity_letters.pdf')
plt.close()


for i in range(6):
	print(ttest_ind(davg[i,0,:], davg[i,1,:]))
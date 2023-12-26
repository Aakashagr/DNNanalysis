# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:08:21 2022

@author: Aakash
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import os
import torch
import clean_cornets
from scipy.spatial.distance import pdist
import torchvision.transforms as transforms
from torchvision import datasets

#%% extracting 
data_dir = 'stimuli/bigrams_TM/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])
stim, classes = next(dataiter)

#%%


dmat = np.zeros((12,2,5))

for rep in range(5): 
	feat = {};
	feat['T'] = {}; feat['M'] = {}; 
	
	for i in ['T','M']: 
		if i == 'T':
			net = clean_cornets.CORNet_Z_nonbiased_words()
			checkpoint = torch.load('model_all-lang/save_lit_tel_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
		elif i == 'M':
			net = clean_cornets.CORNet_Z_nonbiased_words()
			checkpoint = torch.load('model_all-lang/save_lit_mal_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
		
		
		for key in list(checkpoint.keys()):
			if 'module.' in key:
				checkpoint[key.replace('module.', '')] = checkpoint[key]
				del checkpoint[key]
		net.load_state_dict(checkpoint)
	
		feat[i]['v1'], feat[i]['v2'], feat[i]['v4'], feat[i]['it'], feat[i]['h'], feat[i]['out'] = net(torch.tensor(stim).float())
	
	# comparing dissimilarities across different cnn and dpsy
	dall = {}
	dall['mal'] = {}; dall['tel'] = {}
	
	for lang in ['mal','tel']:
		if lang == 'mal': 	# Malayalam
			cid = np.arange(25,50); bid = np.arange(300,600)
		else: #Telugu
			cid = np.arange(0,25); bid = np.arange(0,300)
	
		for lay in feat['T'].keys():
			dall[lang][lay] = []
			for cx in feat.keys():
				dall[lang][lay].append(pdist(feat[cx][lay].detach().numpy()[cid,:],metric = 'correlation'))
			
	
	n = 0
	for lang in ['mal','tel']:
		for i,lay in enumerate(dall[lang].keys()):
			dmat[n,:,rep] = np.nanmean(dall[lang][lay],1)
			n +=1

#%%

	
plt.figure()
plt.bar(np.arange(6)-.2,np.mean(dmat[:6,0,:],1), yerr = np.std(dmat[:6,0,:],1), width =0.1)
plt.bar(np.arange(6)-.1,np.mean(dmat[6:,0,:],1), yerr = np.std(dmat[6:,0,:],1), width =0.1)
plt.bar(np.arange(6)+.1,np.mean(dmat[:6,1,:],1), yerr = np.std(dmat[:6,1,:],1), width =0.1)
plt.bar(np.arange(6)+.2,np.mean(dmat[6:,1,:],1), yerr = np.std(dmat[6:,1,:],1), width =0.1)
plt.ylim([0,0.6])
plt.xticks(range(6),labels = ['v1','v2','v4','it','h','out'], size = 16)
plt.ylabel('Mean dissimilarity')
plt.legend(['Malayalam bigrams','Telugu bigrams'])
plt.savefig('manuscript_codes/Dissimilarity_TelMal.pdf')
plt.close()

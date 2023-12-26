# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:09:37 2022

@author: Aakash
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from telNet import clean_cornets
from PIL import Image
from scipy.spatial.distance import pdist, squareform

#%% extracting data from matfiles
import mat73
matfile = mat73.loadmat('L2_letters.mat')['L2_str']

# Stimuli
allstim = matfile['images']
stim = []
for i in range(len(allstim)):
	im = Image.fromarray(np.uint8(255-allstim[i][0]))
	im = im.resize([76,60])
	temp = np.ones((224,224))*255
	temp[82:142,74:150] = im
	stim.append([temp, temp, temp])
stim = np.array(stim)

# RT data
RT = matfile['RT']
ismal = matfile['subjinfo']['ismalayalam']
RT = np.delete(RT, [np.array([12,28,29])-1], axis = 1)
ismal = np.delete(ismal, [np.array([12,28,29])-1])

# Removing outliers in RT data
for i in range(len(RT)):
	RT[i][np.where(RT[i] > RT[i] + 3*np.nanstd(RT[i]))] = np.nan
	
#%%

feat = {};
feat['I'] = {};  feat['T'] = {}; feat['M'] = {}; 
 
for i in ['T','M','I']: 
	if i == 'T':
		net = clean_cornets.CORNet_Z_nonbiased_words()
		checkpoint = torch.load('telNet/save/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	elif i == 'M':
		net = clean_cornets.CORNet_Z_nonbiased_words()
		checkpoint = torch.load('malNet/save/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	else:
		net = clean_cornets.CORnet_Z_tweak()
		checkpoint = torch.load('telNet/save/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	
	
	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)

	feat[i]['v1'], feat[i]['v2'], feat[i]['v4'], feat[i]['it'], feat[i]['h'], feat[i]['out'] = net(torch.tensor(stim).float())

#%% comparing dissimilarities across different cnn and dpsy
dall = {}
dall['mal'] = {}; dall['tel'] = {}

for lang in ['mal','tel']:
	if lang == 'mal': 	# Malayalam
		cid = np.arange(0,36); bid = np.arange(0,630)
	else: #Telugu
		cid = np.arange(36,72); bid = np.arange(630,1260)

	for lay in feat['T'].keys():
		dall[lang][lay] = []
		dall[lang][lay].append(1./np.nanmean(np.nanmean(RT[:,ismal == 0,:],2),1)[bid])
		dall[lang][lay].append(1./np.nanmean(np.nanmean(RT[:,ismal == 1,:],2),1)[bid])
		
		for cx in feat.keys():
			dall[lang][lay].append(pdist(feat[cx][lay].detach().numpy()[cid,:],metric = 'correlation'))
		

#%%
for lang in ['mal','tel']:
	fig = plt.figure(figsize=(20, 3))	
	for i,lay in enumerate(dall[lang].keys()):
		plt.subplot(1,6,i+1)
		rmat = 1-squareform(pdist(dall[lang][lay],'correlation'))
		im = plt.imshow(rmat)
		plt.clim(0.3,1);
		
	fig.subplots_adjust(right=0.9)	
	cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	

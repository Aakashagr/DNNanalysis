# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:41:32 2022

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
from torchvision import datasets, transforms

#%% extracting data from matfiles
from scipy.io import loadmat
matfile = loadmat('dfmri.mat')

# Stimuli
# allstim = matfile['images']
# stim = []
# for i in range(len(allstim)):
# 	im = Image.fromarray(np.uint8(255-allstim[i][0]))
# 	xscale = int(60/225*np.shape(im)[1])
# 	im = im.resize([xscale,60])
# 	temp = np.ones((224,224))*255
# 	temp[82:142,112-int(xscale/2):112+int(xscale/2)] = im
# 	stim.append([temp, temp, temp])
# stim = np.array(stim)

# dissimilarity data
ismal = matfile['ismal'].flatten()
Tdis =  matfile['Tdis']
Mdis =  matfile['Mdis']


data_dir = 'stimuli/bigrams_fmri/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])
stim, classes = next(dataiter)


#%%

feat = {};
feat['I'] = {};  feat['T'] = {}; feat['M'] = {}; 
 
for i in ['I','T','M']: 
	if i == 'T':
		net = clean_cornets.CORNet_Z_nonbiased_words()
		checkpoint = torch.load('telNet/save_rep1/save_lit_no_bias_z_77_full_nomir.pth.tar',map_location ='cpu')['state_dict']
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
		cid = np.arange(0,24); 
	else: #Telugu
		cid = np.arange(24,48); 
		
	for lay in feat['T'].keys():
		dall[lang][lay] = []
		for cx in feat.keys():
			dall[lang][lay].append(pdist(feat[cx][lay].detach().numpy()[cid,:],metric = 'correlation'))
		

#%% Mean dissimilarity

dmat = np.zeros((12,3))
n = 0
for lang in ['mal','tel']:
	for i,lay in enumerate(dall[lang].keys()):
		dmat[n,:] = np.nanmean(dall[lang][lay],1)
		n +=1
	
fig = plt.figure(figsize=(35, 5))	
for i in range(6):
	plt.subplot(1,6,i+1)
	plt.bar(np.arange(0,3)-0.1, dmat[i,:], width = 0.2)
	plt.bar(np.arange(0,3)+0.1, dmat[i+6,:], width = 0.2)
	plt.title(list(dall[lang].keys())[i], size = 30)
	plt.xticks(range(3),labels = ['illi cnn','tel cnn','mal cnn'], 
								rotation = 45, size = 20)
	plt.ylabel('Mean dissimilarity', size=20)
	plt.ylim([0,0.7])
	
plt.legend(['Malayalam bigrams','Telugu bigrams'])
#%% comparing behaviour dissimilarity between fmri and cnn
from scipy.stats import spearmanr, zscore

lang = 'tel'
cnn_id = 1
rmat_t = np.zeros((5,6))
for i,lay in enumerate(feat['T'].keys()):
	for roi in range(len(Mdis)):
		rmat_t[roi,i] = spearmanr(dall[lang][lay][cnn_id], np.nanmean(Tdis[roi, ismal == 0,:],0))[0]


lang = 'mal'
cnn_id = 2
rmat_m = np.zeros((5,6))
for i,lay in enumerate(feat['T'].keys()):
	for roi in range(len(Mdis)):
		rmat_m[roi,i] = spearmanr(dall[lang][lay][cnn_id], np.nanmean(Mdis[roi, ismal == 1,:],0))[0]



rmat_all = np.zeros((5,6))
for i,lay in enumerate(feat['T'].keys()):
	for roi in range(len(Mdis)):
		rmat_all[roi,i] = spearmanr(np.r_[zscore(dall['mal'][lay][2]),  zscore(dall['tel'][lay][1])],
							   np.r_[zscore(np.nanmean(Mdis[roi, ismal == 1,:],0)), zscore(np.nanmean(Tdis[roi, ismal == 0,:],0))])[0]
# 		rmat_all[roi,i] = spearmanr(np.r_[(dall['mal'][lay][2]),  (dall['tel'][lay][1])],
# 					   np.r_[(np.nanmean(Mdis[roi, ismal == 1,:],0)), (np.nanmean(Tdis[roi, ismal == 0,:],0))])[0]



#%%
plt.figure()
plt.plot(rmat_t.T)
plt.xticks(range(6), labels = feat['T'].keys())
plt.legend(['V1-V3','V4','LO','VWFA','TG'])
plt.title('Telugu stimuli')

plt.figure()
plt.plot(rmat_m.T)
plt.xticks(range(6), labels = feat['T'].keys())
plt.legend(['V1-V3','V4','LO','VWFA','TG'])
plt.title('Malayalam stimuli')

# plt.figure()
# plt.plot(rmat_all.T)
# plt.xticks(range(6), labels = feat['T'].keys())
# plt.legend(['V1-V3','V4','LO','VWFA','TG'])
# plt.title('Malayalam stimuli')
#%% Split half consistency of the data

Tsub = np.where(ismal==0)[0]	
Msub = np.where(ismal==1)[0]

sh_TT = []; sh_TM = []
sh_MT = []; sh_MM = []

for roi in range(5):
	sh_TT.append(spearmanr(np.nanmean(Tdis[roi, Tsub[::2],:],0),np.nanmean(Tdis[roi,Tsub[1::2],:],0))[0])
	sh_TM.append(spearmanr(np.nanmean(Mdis[roi, Tsub[::2],:],0),np.nanmean(Mdis[roi,Tsub[1::2],:],0))[0])
	sh_MT.append(spearmanr(np.nanmean(Tdis[roi, Msub[::2],:],0),np.nanmean(Tdis[roi,Msub[1::2],:],0))[0])
	sh_MM.append(spearmanr(np.nanmean(Mdis[roi, Msub[::2],:],0),np.nanmean(Mdis[roi,Msub[1::2],:],0))[0])
	

plt.figure()
plt.bar(np.arange(0,5)-0.3, sh_TT, width = 0.2)
plt.bar(np.arange(0,5)-0.1, sh_TM, width = 0.2)
plt.bar(np.arange(0,5)+0.1, sh_MT, width = 0.2)
plt.bar(np.arange(0,5)+0.3, sh_MM, width = 0.2)
plt.xticks(range(5), labels = ['V1-V3','V4','LO','VWFA','TG'])
plt.ylabel('Correlation coefficient')
plt.legend(['TT (sub:stim)','TM','MT','MM'])

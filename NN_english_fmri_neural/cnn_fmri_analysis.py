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
from training import clean_cornets
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from torchvision import datasets, transforms
from scipy.stats import spearmanr, pearsonr

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
dpsyn = matfile['dpsyn'].flatten()
dpsyw =  matfile['dpsyw'].flatten()
dpsynw =  matfile['dpsynw'].flatten()
droi =  matfile['droi']
dsem =  matfile['dsemfeat'].flatten()
sid =  matfile['sid'].flatten()
nid =  matfile['nid'].flatten()
wid =  matfile['wid'].flatten()
nwid =  matfile['nwid'].flatten()
meanRTn =  matfile['meanRTn'].flatten()
meanRTw =  matfile['meanRTw'].flatten()


data_dir = 'stimuli/words/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}

rpsy = np.zeros((6,2,5,2))

for rep in range(5):
	print(rep)
	dataiter = iter(dataloaders['train'])
	stim, classes = next(dataiter)
	
	feat = {};
	feat['I'] = {};  feat['E'] = {}; 
	 
	for i in ['I','E']: 
		if i == 'E':
			net = clean_cornets.CORNet_Z_nonbiased_words()
			checkpoint = torch.load('training/save/save_lit_en_rep'+str(rep)+'.pth.tar',map_location ='cpu')['state_dict']
		else:
			net = clean_cornets.CORnet_Z_tweak()
			checkpoint = torch.load('training/save/save_lit_illit_rep'+str(rep)+'.pth.tar',map_location ='cpu')['state_dict']
		
		
		for key in list(checkpoint.keys()):
		    if 'module.' in key:
		        checkpoint[key.replace('module.', '')] = checkpoint[key]
		        del checkpoint[key]
		net.load_state_dict(checkpoint)
	
		feat[i]['v1'], feat[i]['v2'], feat[i]['v4'], feat[i]['it'], feat[i]['h'], feat[i]['out'] = net(torch.tensor(stim).float())
	
	#% comparing dissimilarities across different cnn and dpsy
	dall = {}
	dall['w'] = {}; dall['n'] = {}; dall['s'] = {}
	idx = {}
	idx['w'] = range(32); idx['n'] = np.arange(32,64); idx['s'] = range(64)
	
	for stype in dall.keys():
		for lay in feat['E'].keys():
			dall[stype][lay] = []
			for cx in feat.keys():
				dall[stype][lay].append(pdist(feat[cx][lay].detach().numpy()[idx[stype],:],metric = 'correlation'))
			
	#% Comparing cnn and behaviour dissimilarity
	
	for sidx, stype in enumerate(['n','w']):
		if stype == 'n':
			d = dpsyn
		else:
			d = dpsyw
				
		for cid in range(2):
			for i,lay in enumerate(feat['E'].keys()):
				rpsy[i,cid,rep,sidx] = pearsonr(dall[stype][lay][cid], d)[0]
			
#%%
for i,tname in enumerate(['nonwords','words']):
	plt.figure()	
	plt.plot(np.mean(rpsy[:,:,:,i],2))
# 	for j in range(2): # Literate and illiterate network
# 		yerr = np.std(rpsy[:,j,:,i],1)
# 		yavg = np.mean(rpsy[:,j,:,i],1)
# 		plt.plot(yavg)
# 		plt.fill_between(range(6), yavg-yerr, yavg+yerr, alpha= .50)
	plt.xticks(range(6), labels = feat['E'].keys())
	plt.legend(['Illiterate n/w','Literate n/w'])
	plt.title(tname)
# 	plt.ylim([0.15,0.5])
	
#%% Mean dissimilarity

if 0:
	dmat = np.zeros((18,2))
	n = 0
	for stype in dall.keys():
		for i,lay in enumerate(dall[stype].keys()):
			dmat[n,:] = np.nanmean(dall[stype][lay],1)
			n +=1
		
	fig = plt.figure(figsize=(35, 5))	
	for i in range(6):
		plt.subplot(1,6,i+1)
		plt.bar(np.arange(0,2)-0.2, dmat[i,:], width = 0.2)
		plt.bar(np.arange(0,2), dmat[i+6,:], width = 0.2)	
		plt.bar(np.arange(0,2)+0.2, dmat[i+12,:], width = 0.2)
		plt.title(list(dall[stype].keys())[i], size = 30)
		plt.xticks(range(2),labels = ['illi cnn','Eng cnn'],rotation = 45, size = 20)
		plt.ylabel('Mean dissimilarity', size=20)
		plt.ylim([0,0.8])
		
	plt.legend(['words','nonwords','all'])

#%% comparing behaviour dissimilarity between fmri and cnn

if 0:
	for stype in ['n','w','s']:
		if stype == 'n':
			tname = 'nonwords'
			d = nid
		elif stype == 's':
			tname = 'all stimuli'
			d = sid
		else:
			tname = 'words'
			d = wid
			
		cnn_id = 1
		rmat = np.zeros((5,6))
		for i,lay in enumerate(feat['E'].keys()):
			for roi in range(5):
				rmat[roi,i] = spearmanr(dall[stype][lay][cnn_id], np.nanmedian(droi[d,:,roi],1))[0]
		
		#%
		plt.figure()
		plt.plot(rmat.T)
		plt.xticks(range(6), labels = feat['E'].keys())
		plt.legend(['V1-V3','V4','LO','VWFA','TG'])
		plt.title(tname)


#%% Split half consistency of the data
if 0:
	sh = [];
	for roi in range(5):
		sh.append(spearmanr(np.nanmean(droi[sid,::2,roi],1),np.nanmean(droi[sid,1::2,roi],1))[0])	
	
	plt.figure()
	plt.bar(np.arange(0,5), sh, width = 0.2)
	plt.xticks(range(5), labels = ['V1-V3','V4','LO','VWFA','TG'])
	plt.ylabel('Correlation coefficient')
	plt.title('Split-half consistency')
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:27:27 2022

@author: Aakash
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch
from telNet import clean_cornets
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import torchvision.transforms as transforms
from torchvision import datasets, transforms

#%% extracting data from matfiles
import mat73
matfile = mat73.loadmat('L2_telmal_bigram.mat')['L2_str']

# Stimuli
# allstim = matfile['images']
# stim = []
# for i in range(len(allstim)):
# 	im = Image.fromarray(np.uint8(255-allstim[i][0]))
# 	im = im.resize([114,60])
# 	temp = np.ones((224,224))*255
# 	temp[82:142,55:169] = im
# 	stim.append([temp, temp, temp])
# stim = np.array(stim)

# RT data
RT = matfile['RT']
ismal = matfile['subjinfo']['ismalayalam']

# Removing outliers in RT data
for i in range(len(RT)):
	RT[i][np.where(RT[i] > RT[i] + 3*np.nanstd(RT[i]))] = np.nan
	

data_dir = 'stimuli/bigrams/'
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
		checkpoint = torch.load('telNet/save_rep2/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	elif i == 'M':
		net = clean_cornets.CORNet_Z_nonbiased_words()
		checkpoint = torch.load('malNet/save_rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	else:
		net = clean_cornets.CORnet_Z_tweak()
		checkpoint = torch.load('telNet/save_rep1/save_pre_z_49_full_nomir.pth.tar',map_location ='cpu')['state_dict']
	
	
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
		cid = np.arange(25,50); bid = np.arange(300,600)
	else: #Telugu
		cid = np.arange(0,25); bid = np.arange(0,300)

	for lay in feat['T'].keys():
		dall[lang][lay] = []
		dall[lang][lay].append(1./np.nanmean(np.nanmean(RT[:,ismal == 0,:],2),1)[bid])
		dall[lang][lay].append(1./np.nanmean(np.nanmean(RT[:,ismal == 1,:],2),1)[bid])
		
		for cx in feat.keys():
			dall[lang][lay].append(pdist(feat[cx][lay].detach().numpy()[cid,:],metric = 'correlation'))
		

#%% representation similarity
for lang in ['mal','tel']:
	fig = plt.figure(figsize=(20, 4))	
	for i,lay in enumerate(dall[lang].keys()):
		plt.subplot(1,6,i+1)
		rmat = 1-squareform(pdist(dall[lang][lay],'correlation'))
		print(rmat[-2,0])
		im = plt.imshow(rmat)
		plt.clim(0,1);
		plt.title(lay, size = 30)
		plt.xticks(range(5),labels = ['tel lit','mal lit','illi cnn',
								'tel cnn','mal cnn'], rotation = 45, size=15)
		
		
	plt.suptitle(lang+' stimuli', size = 30)
	fig.subplots_adjust(right=0.9)	
	cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	
#%% Mean dissimilarity

dmat = np.zeros((12,5))
n = 0
for lang in ['mal','tel']:
	for i,lay in enumerate(dall[lang].keys()):
		dmat[n,:] = np.nanmean(dall[lang][lay],1)
		n +=1
	
fig = plt.figure(figsize=(35, 5))	
for i in range(6):
	plt.subplot(1,6,i+1)
	plt.bar(np.arange(0,3)-0.1, dmat[i,2:], width = 0.2)
	plt.bar(np.arange(0,3)+0.1, dmat[i+6,2:], width = 0.2)
	plt.title(list(dall[lang].keys())[i], size = 30)
	plt.xticks(range(3),labels = ['illi cnn','tel cnn','mal cnn'], 
								rotation = 45, size = 20)
	plt.ylabel('Mean dissimilarity', size=20)
	plt.ylim([0,0.7])
	
plt.legend(['Malayalam bigrams','Telugu bigrams'])

#%% Building compositional part-sum model

import itertools
from sklearn.linear_model import LinearRegression
sid = [p for p in itertools.product(range(5), repeat=2)] # stimuli index
lid = np.array(list(itertools.combinations(range(5), 2))) # Letter pair index
bid = list(itertools.combinations(range(25), 2)) # bigram pair index

Xmat = np.zeros((len(bid),len(lid)*3))   
for i in range(np.shape(Xmat)[0]):
	s1 = sid[bid[i][0]]
	s2 = sid[bid[i][1]]
	
	c1 = np.where(np.all(lid == np.sort([s1[0],s2[0]]),axis=1))[0]
	c2 = np.where(np.all(lid == np.sort([s1[1],s2[1]]),axis=1))[0]
	a1 = np.where(np.all(lid == np.sort([s1[0],s2[1]]),axis=1))[0]
	a2 = np.where(np.all(lid == np.sort([s1[1],s2[0]]),axis=1))[0]
	w1 = np.where(np.all(lid == np.sort([s1[0],s1[1]]),axis=1))[0]
	w2 = np.where(np.all(lid == np.sort([s2[0],s2[1]]),axis=1))[0]
	
	
	if len(c1): 
		Xmat[i,c1[0] + len(lid)*0] += 1
	if len(c2): 
		Xmat[i,c2[0] + len(lid)*0] += 1	 
	if len(a1): 
		Xmat[i,a1[0] + len(lid)*1] += 1
	if len(a2):
		Xmat[i,a2[0] + len(lid)*1] += 1  
	if len(w1): 
		Xmat[i,w1[0] + len(lid)*2] += 1
	if len(w2):
		Xmat[i,w2[0] + len(lid)*2] += 1  
	
#%%  
metric = 'correlation'
lang = 'tel'
r2 = np.zeros((6,3))
coeff = np.zeros((6,len(lid)*3+1, 3))
for i,l in enumerate(dall['tel'].keys()):
	
	for n in [2,3,4]:
		dx = np.array(dall[lang][l])[n,:]
		reg = LinearRegression().fit(Xmat, dx)
		r2[i,n-2] = reg.score(Xmat, dx)
		coeff[i,:,n-2] = np.r_[reg.coef_, reg.intercept_]
	
	
plt.figure()
plt.bar(np.arange(6)-.2,r2[:,0], width =0.2)
plt.bar(np.arange(6),r2[:,1], width =0.2)
plt.bar(np.arange(6)+.2,r2[:,2], width =0.2)
plt.xticks(range(6),labels = dall['tel'].keys(), size = 16)
plt.ylabel('Model fit R2')
plt.legend(['illiterate n/w','Telugu n/w','Malayalam n/w'], loc = 'lower left')
plt.title(lang + ' stimuli')

fig = plt.figure(figsize=(35, 5))	
# Plotting the coeff
for i,l in enumerate(dall['tel'].keys()):
	cili = np.mean(np.reshape(coeff[i,:-1,0],[3,10]),1)
	ctel = np.mean(np.reshape(coeff[i,:-1,1],[3,10]),1)
	cmal = np.mean(np.reshape(coeff[i,:-1,2],[3,10]),1)
	
	plt.subplot(1,6,i+1)
	plt.bar(np.arange(3)-.2,cili, width =0.2)
	plt.bar(np.arange(3),ctel, width =0.2)
	plt.bar(np.arange(3)+.2,cmal, width =0.2)
	plt.xticks(range(3),labels = ['C','A','W'], size = 16)
	plt.ylabel('Model coeff')
	plt.title(l, size = 20)

plt.legend(['illiterate n/w','Telugu n/w','Malayalam n/w'])
plt.suptitle(lang + ' stimuli')


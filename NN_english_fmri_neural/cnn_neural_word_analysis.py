# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:03:23 2022

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
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr

#%%
f = np.load('CNN_comparison.npz')
data = f['data']
coords = f['coords']
wordlist = f['wordlist']
wlen = np.array([len(i) for i in wordlist])
wlen_id = np.where(wlen > 3)[0]

from sklearn.cluster import KMeans
nc = 5
kmeans = KMeans(n_clusters=nc, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_cat = kmeans.fit_predict(coords)

for i in range(nc):
    print(len(np.where(cluster_cat == i)[0]))
	
from nilearn import plotting
plotting.plot_markers(cluster_cat, coords,node_size = 10, display_mode='lyrz')

#%% Visualizing RDMs for each cluster
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import string
strlist = list(string.ascii_uppercase)  # storing charaacters from A-Z

pca = PCA(n_components = 2)
mds = MDS(n_components = 2, metric = False, dissimilarity= 'precomputed')
resp = (np.mean(data[cluster_cat == 0,:,:10],2).T)[wlen_id,:]
# resp = feat['E']['it'].detach().numpy()

X_embedded = mds.fit_transform(squareform(pdist(resp,'correlation')), init = pca.fit_transform(resp))
plt.figure(figsize = [10,10])
for i in range(len(wordlist[wlen_id])):
		plt.text(X_embedded[i,0] ,X_embedded[i,1], wordlist[wlen_id[i]], size = 15)
	
lim = [np.min(X_embedded[:]) ,np.max(X_embedded[:])]
plt.xlim(lim)
plt.ylim(lim)
plt.title('Cluster 0, (250-500 ms)')

print(spearmanr(pdist(X_embedded), pdist(resp,'correlation')))
#%% extracting data from matfiles

data_dir = 'stimuli/words_oscar/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}

rmat = np.zeros((6,nc,5,2)); pmat = np.zeros((6,nc,5,2))
rmati = np.zeros((6,nc,5,2)); pmati = np.zeros((6,nc,5,2))

for rep in range(5):
	dataiter = iter(dataloaders['train'])
	stim, classes = next(dataiter)
	print(rep)
	
	feat = {}; feat['I'] = {}; feat['E'] = {}; 
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


	#% cluster level analysis
	for i,l in enumerate(feat['E'].keys()):
		for cc in range(nc):
			rmat[i,cc,rep,0],pmat[i,cc,rep,0] = spearmanr(pdist(feat['E'][l][wlen_id,:].detach().numpy(),'correlation'),
									pdist((np.mean(data[cluster_cat == cc,:,:10],2).T)[wlen_id,:],'correlation'))
			rmat[i,cc,rep,1],pmat[i,cc,rep,1] = spearmanr(pdist(feat['E'][l][wlen_id,:].detach().numpy(),'correlation'),
									pdist((np.mean(data[cluster_cat == cc,:,10:],2).T)[wlen_id,:],'correlation'))

	for i,l in enumerate(feat['I'].keys()):
		for cc in range(nc):
			rmati[i,cc,rep,0],pmati[i,cc,rep,0] = spearmanr(pdist(feat['I'][l][wlen_id,:].detach().numpy(),'correlation'),
									pdist((np.mean(data[cluster_cat == cc,:,:10],2).T)[wlen_id,:],'correlation'))
			rmati[i,cc,rep,1],pmati[i,cc,rep,1] = spearmanr(pdist(feat['I'][l][wlen_id,:].detach().numpy(),'correlation'),
									pdist((np.mean(data[cluster_cat == cc,:,10:],2).T)[wlen_id,:],'correlation'))
	
	
#%%
				
for i, tp in enumerate(['(0-250 ms)','(250-500 ms)']):
	plt.figure()
	plt.plot(np.mean(rmat[:,:,:,i],2))
	plt.legend(['Cluster: ' + str(i) for i in range(nc)])
	plt.title('Match to neural data - oscar  '+tp)
	plt.xticks(range(6), labels = feat['E'].keys())
	plt.ylabel('Correlation coefficient')

				
#%% Unit level analysis

if 0:
	nu = np.shape(data)[0]
	rmat = np.zeros((nu,6))
	pmat = np.zeros((nu,6))
	for i,l in enumerate(feat['E'].keys()):
		for cc in range(nu):
			rmat[cc,i],pmat[cc,i] = spearmanr(pdist(feat['E'][l].detach().numpy(),'correlation'),
									pdist(np.expand_dims(np.mean(data[cc,:,10:],2).T,axis=1)))
	
# 	plt.figure()
# # 	plt.plot(rmat.T)
# 	# plt.legend(range(nu))
# 	plt.title('Match to neural data')
# 	plt.xticks(range(6), labels = feat['E'].keys())
	
	#%
	plotting.plot_markers(rmat[:,4] - rmat[:,2], coords,node_size = 10, display_mode='lyrz'
						  ,node_cmap= 'bwr',node_vmin = -.051, node_vmax = .051,)


#%%
if 0:
	rcnn = np.zeros((6,6))
	for i,l1 in enumerate(feat['E'].keys()):
		for j,l2 in enumerate(feat['E'].keys()):
			rcnn[i,j],_ = spearmanr(pdist(feat['E'][l1].detach().numpy(),'correlation'),
							  pdist(feat['E'][l2].detach().numpy(),'correlation'))
			
		
	plt.figure()
	plt.imshow(rcnn)
	plt.xticks(range(6), labels = feat['E'].keys())
	plt.yticks(range(6), labels = feat['E'].keys())
	plt.colorbar()
	plt.title('Across 22 word pairs')
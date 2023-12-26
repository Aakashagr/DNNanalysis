# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:01:40 2022

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

coords = loadmat('Jabberworcky_metadata.mat')['coords']
dis_beh = loadmat('dis_uletter.mat')['dis_uletter']
popt_slm = np.load('letter_model_param_lasso.npz', allow_pickle = True)['popt_slm']
rfit_slm = np.load('letter_model_param_lasso.npz', allow_pickle = True)['rfit_slm']


from sklearn.cluster import KMeans
nc = 4
kmeans = KMeans(n_clusters=nc, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_cat = kmeans.fit_predict(coords)

for i in range(nc):
    print(len(np.where(cluster_cat == i)[0]))
	
from nilearn import plotting
plotting.plot_markers(cluster_cat, coords,node_size = 10, display_mode='lyrz')

plotting.plot_markers(np.mean(rfit_slm[:,10:],1), coords,node_size = 10, node_vmax = .5,
					  display_mode='lyrz',title = 'model fit (250-500 ms)')


let_freq = np.log([8.04, 1.48, 3.34, 3.82, 12.49, 2.4, 1.87, 5.05, 7.57, 0.16, 0.54, 4.07, 2.51, 7.23, 7.64, 2.14, 0.12, 6.28, 6.51, 9.28, 2.73, 1.05, 1.68, 0.23, 1.66, 0.09])
dlet = pdist(np.expand_dims(let_freq, axis = 1))
for cc in range(nc):
 	plt.figure()
 	r,p = spearmanr(dlet, pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,10:,:-1,:],3),1).T, metric = 'euclidean'))
 	plt.scatter(dlet, pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,10:,:-1,:],3),1).T, metric = 'euclidean'))
 	 	
	 # r,p = spearmanr(dis_beh, pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,:10,:-1,:],3),1).T, metric = 'euclidean'))
 	# plt.scatter(dis_beh, pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,:10,:-1,:],3),1).T, metric = 'euclidean'))
 	plt.title('cluster = '+ str(cc)+', Corr. coef = ' + str(r.round(2))+ ', p = '+ str(p.round(3)))
 	# plt.ylabel('Estimated dissimilarity'); plt.xlabel('Behaviour dissimilarity')
 	plt.ylabel('Estimated dissimilarity'); plt.xlabel('log letter frequency dissimilarity')

#%% Visualizing RDMs for each cluster
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import string
strlist = list(string.ascii_uppercase)  # storing charaacters from A-Z

pca = PCA(n_components = 2)
mds = MDS(n_components = 2, metric = False, dissimilarity= 'precomputed')
# resp = np.mean(np.mean(popt_slm[cluster_cat == 0,10:,:-1,:],3),1).T
resp = feat['E']['it'].detach().numpy()

X_embedded = mds.fit_transform(squareform(pdist(resp,'correlation')), init = pca.fit_transform(resp))
plt.figure(figsize = [10,10])
for i in range(26):
		plt.text(X_embedded[i,0] ,X_embedded[i,1], strlist[i].upper(), size = 15)
	
lim = [np.min(X_embedded[:]) ,np.max(X_embedded[:])]
plt.xlim(lim)
plt.ylim(lim)
plt.title('Cluster 0, (250-500 ms)')
#%% extracting data from matfiles

data_dir = 'stimuli/letters/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}

rmat = np.zeros((6,nc,5,2))
pmat = np.zeros((6,nc,5,2))
rpsy = np.zeros((6,2,5))

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
			rmat[i,cc,rep,0],pmat[i,cc,rep,0] = pearsonr(pdist(feat['E'][l].detach().numpy(),'correlation'),
									pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,:10,:-1,:],3),1).T))
			rmat[i,cc,rep,1],pmat[i,cc,rep,1] = pearsonr(pdist(feat['E'][l].detach().numpy(),'correlation'),
									pdist(np.mean(np.mean(popt_slm[cluster_cat == cc,10:,:-1,:],3),1).T))

	
	#% Match to behaviour
	for i,l in enumerate(feat['E'].keys()):
		for j,t in enumerate(feat.keys()):
# 			rpsy[i,j,rep],_ = pearsonr(pdist(feat[t][l].detach().numpy(),'correlation'),loadmat('dis_uletter.mat')['dis_uletter'])
			rpsy[i,j,rep],_ = pearsonr(pdist(feat[t][l].detach().numpy(),'correlation'),dlet)

#%%
				
for i, tp in enumerate(['(0-250 ms)','(250-500 ms)']):
	plt.figure()
	plt.plot(np.mean(rmat[:,:,:,i],2))
	plt.legend(['Cluster: ' + str(i) for i in range(nc)])
	plt.title('Match to neural data, oscar model parameters '+tp)
	plt.xticks(range(6), labels = feat['E'].keys())
	plt.ylabel('Correlation coefficient')

				
plt.figure()
plt.plot(np.mean(rpsy,2))
plt.title('Match to letter dissimilarity, dpsy')
plt.xticks(range(6), labels = feat['E'].keys())
plt.ylabel('Correlation coefficient')
plt.legend(['Illiterate n/w','Literate n/w'])
#%% Unit level analysis

if 1:
	nu = np.shape(popt_slm)[0]
	rmat = np.zeros((nu,6))
	pmat = np.zeros((nu,6))
	for i,l in enumerate(feat['E'].keys()):
		for cc in range(nu):
			rmat[cc,i],pmat[cc,i] = spearmanr(pdist(feat['E'][l].detach().numpy(),'correlation'),
									pdist(np.expand_dims(np.mean(np.mean(popt_slm[cc,10:,:-1,:],2),0).T,axis=1)))
	
# 	plt.figure()
# # 	plt.plot(rmat.T)
# 	# plt.legend(range(nu))
# 	plt.title('Match to neural data')
# 	plt.xticks(range(6), labels = feat['E'].keys())
	
	#%%
	plotting.plot_markers(rmat[:,4] - rmat[:,2], coords,node_size = 10, display_mode='lyrz'
						  ,node_cmap= 'bwr',node_vmin = -.051, node_vmax = .051,)


#%%
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

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:17:18 2021

@author: Aakash
"""
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import torchvision
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat
import os, sys
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
import clean_cornets

#%%
from torchvision import datasets, transforms

data_dir = '../stimuli/HierStim/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}

qo = np.array(np.arange(0,400));
qe = np.array(np.arange(400,1660));
qc = np.array(np.arange(1660,2920));
#%%

meanresp = np.zeros((18,3,5))
for r in range(5):
	rep = 'rep'+str(r+1); print(rep)
	nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
	nBli['it'] = []; nBli['h'] = []; nBli['out'] = []

	net = clean_cornets.CORNet_Z_nonbiased_words()
	checkpoint = torch.load(rep+'/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']

	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)
	net.eval()

	dataiter = iter(dataloaders['train'])
	for i in range(37):
	    stimtemp, classes = next(dataiter)
	    varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
	    nBli['v1'].extend(varV1.detach().numpy())
	    nBli['v2'].extend(varV2.detach().numpy())
	    nBli['v4'].extend(varV4.detach().numpy())
	    nBli['it'].extend(varIt.detach().numpy())
	    nBli['h'].extend(varh.detach().numpy())
	    nBli['out'].extend(varOut.detach().numpy())
	    print(i)

	#%%
	from WSunits import wordsel_units
	neuid_e2, neuid_c2, neuid_ec = wordsel_units(rep)
	nid = {}
	nid[0] = neuid_e2
	nid[1] = neuid_c2
	nid[2]= list(set(neuid_ec)^set(neuid_c2)^set(neuid_e2))
	type = ['English','Chinese','English+Chinese']

	data_d = np.array(nBli['h'])
	for i in range(3):
	    for t in range(4):
	        meanresp[t,i,r] = (np.nanmean(np.nanmean(data_d[np.arange(t*100,(t+1)*100), np.reshape(nid[i],[np.size(nid[i]),1])  ],1),0))
	    for t in range(14):
	        meanresp[t+4,i,r] = (np.nanmean(np.nanmean(data_d[np.arange(400+t*180,400+(t+1)*180),np.reshape(nid[i],[np.size(nid[i]),1])],1),0))



#%% Plotting the mean response averaged across all units
color = loadmat('../7TbilingualCE_colorcode.mat')['bilingualCEcolormap'][:,1:]
color = np.concatenate(([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],color))

label = ['bodies','faces','houses','tools','L-','L+','B-','B+','Q-','Q+',
          'E_words', 'stroke','Nrad','Nchar1','Nchar2','NwReal','LF','HF']

fig, axs = plt.subplots(1,3, figsize=(30,5), facecolor='w', edgecolor='k')
axs = axs.ravel();

for i in range(3):
	axs[i].bar(range(18),np.mean(meanresp[:,i,:],1),yerr = np.std(meanresp[:,i,:],1), color = color)
	axs[i].set_xticks(range(18)); axs[i].set_xticklabels(label,rotation = 45)
	axs[i].set_ylabel('Mean response')
	axs[i].set_title(type[i]+ ' word seletive units')


plt.savefig('mean_resp.pdf')
#%% Plotting the mean response for each unit in h layer
if 0:
	fig, axs = plt.subplots(8,5, figsize=(30,40), facecolor='w', edgecolor='k')
	axs = axs.ravel();

	color = ['k','k','k','k','b','c','green','limegreen','orange','gold','tomato','steelblue','skyblue','slateblue','orchid','lightcoral','r','maroon']
	for n,nid in enumerate(np.sort(list(set(neuid_c)^set(neuid_ec)))):
	# for n,nid in enumerate(np.sort(neuid_fc)):
	    meanresp = []
	    for t in range(4):
	        meanresp.append(np.nanmean(data_d[range(t*100,(t+1)*100),nid],0))
	    for t in range(14):
	        meanresp.append(np.nanmean(data_d[range(400+t*180,400+(t+1)*180),nid],0))

	    axs[n].bar(range(18),meanresp,color = color)
	    axs[n].set_title('unit id: ' + str(nid),fontsize = 24)
	fig.suptitle('Chinese selective units', fontsize = 32)


#%%
mean_acc = np.zeros((5,2))
mean_acc[0,0] = 87.064; mean_acc[0,1] = 96.796;
for i in np.arange(1,5):
	acc = np.load('rep'+str(i+1)+'/cat_scores_lit_no_bias_z_full_nomir.npy')
	mean_acc[i,0] = max(np.mean(acc[:,1000:1500],1))
	mean_acc[i,1] = max(np.mean(acc[:,1500:],1))

# plt.plot(np.mean(acc[:,:1000],1))
# plt.plot(np.mean(acc[:,1000:1500],1))
# plt.plot(np.mean(acc[:,1500:],1))
# plt.ylabel('Accuracy'); plt.xlabel('Epoch')
# plt.legend(['Objects','English','Chinese'])
# plt.show()
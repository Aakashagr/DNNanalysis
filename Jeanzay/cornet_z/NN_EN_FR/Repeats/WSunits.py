# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:17:18 2021

@author: Aakash
"""

def wordsel_units(rep):

	import numpy as np
	import scipy
	import matplotlib as mpl
	from matplotlib import pyplot as plt
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

	data_dir = '../stimuli/wordselective_stimuli/'
	transform = {'train': transforms.Compose([transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

	chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
	dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}

	#%%
	dataiter = iter(dataloaders['train'])

	nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
	nBli['it'] = []; nBli['h'] = []; nBli['out'] = []

	literate = 1
	if literate:
		net = clean_cornets.CORNet_Z_nonbiased_words()
		checkpoint = torch.load(rep+'/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
	else:
		net = clean_cornets.CORnet_Z_tweak()
		checkpoint = torch.load('rep2/save_illit_z_79_full_nomir.pth.tar')['state_dict']

	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)
	net.eval()

	for i in range(15):
	    stimtemp, classes = next(dataiter)
	    varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
	    nBli['v1'].extend(varV1.detach().numpy())
	    nBli['v2'].extend(varV2.detach().numpy())
	    nBli['v4'].extend(varV4.detach().numpy())
	    nBli['it'].extend(varIt.detach().numpy())
	    nBli['h'].extend(varh.detach().numpy())
	    nBli['out'].extend(varOut.detach().numpy())
	    print(i)

	#%% Identify word selective units based on 3 standard deviations above the mean of object responses
	qo = np.array(np.arange(0,400));
	qe = np.array(np.arange(400,800));
	qf = np.array(np.arange(800,1200));
	data_d = np.array(nBli['h'])

	neuid_f = []; neuid_e = [];
	for unit in range(np.size(data_d,1)):
	    Enmean = np.mean(data_d[qe,unit])
	    Frmean = np.mean(data_d[qf,unit])
	    Objmean   = np.mean(data_d[qo,unit])
	    Objstdev  = np.var(data_d[qo,unit])**0.5

	    if Enmean >= Objmean + 3*Objstdev:
	        neuid_e.append(unit)

	    if Frmean >= Objmean + 3*Objstdev:
	        neuid_f.append(unit)



	#%% Units selective to either of the languages
	neuid_ef = np.unique(np.concatenate([neuid_f,neuid_e]))
	neuid_f2 = []; neuid_e2 = [];
	for unit in neuid_ef:
		Enmean = np.mean(data_d[qe,unit])
		Frmean = np.mean(data_d[qf,unit])
		Enstdev = np.var(data_d[qe,unit])**0.5
		Frstdev = np.var(data_d[qf,unit])**0.5

		if Enmean >= Frmean + 1*Frstdev:
			neuid_e2.append(unit)

		if Frmean >= Enmean + 1*Enstdev:
			neuid_f2.append(unit)

	print(['% word-selective units: ' + str(np.size(neuid_ef)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_ef))])
	print(['% of English word-selective units: ' + str(np.size(neuid_e2)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_e2))])
	print(['% of French word-selective units: '+ str(np.size(neuid_f2)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_f2))])

	return neuid_e2, neuid_f2, neuid_ef
	#%%
# 	from matplotlib_venn import venn2
# 	venn2(subsets = (len(neuid_f2), len(neuid_e2), len(neuid_ef) - len(neuid_f2) - len(neuid_e2)), set_labels = ('French', 'English'))
# 	plt.title('Number of word selective units')
# 	plt.show()
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
import os, sys
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
import clean_cornets
from torchvision import datasets, transforms

#%%

avg_acc = np.zeros((6,5))
for r in range(5):
	rep = 'rep'+str(r+1); print(rep)

	nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
	nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
	literate = 0
	if literate:
		net = clean_cornets.CORNet_Z_nonbiased_words()
# 		checkpoint = torch.load(rep+'/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
		checkpoint = torch.load('../../NN_EN_FR/Repeats/'+rep+'/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
	else:
		net = clean_cornets.CORnet_Z_tweak()
		checkpoint = torch.load(rep+'/save_pre_z_79_full_nomir.pth.tar')['state_dict']

	for key in list(checkpoint.keys()):
	    if 'module.' in key:
	        checkpoint[key.replace('module.', '')] = checkpoint[key]
	        del checkpoint[key]
	net.load_state_dict(checkpoint)
	net.eval()


# 	data_dir = ['../stimuli/wordselective_stimuli/']
	data_dir = ['../../NN_EN_FR/stimuli/wordselective_stimuli/']

	for ddir in data_dir:
		transform = {'train': transforms.Compose([transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

		chosen_datasets = {x: datasets.ImageFolder(os.path.join(ddir, x), transform =  transform[x]) for x in ['train']}
		dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}
		dataiter = iter(dataloaders['train'])
		for i in range(12):
			stimtemp, classes = next(dataiter)
			if i > 3:  #Ignoring general image categories
			    varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
			    nBli['v1'].extend(varV1.detach().numpy())
			    nBli['v2'].extend(varV2.detach().numpy())
			    nBli['v4'].extend(varV4.detach().numpy())
			    nBli['it'].extend(varIt.detach().numpy())
			    nBli['h'].extend(varh.detach().numpy())
			    nBli['out'].extend(varOut.detach().numpy())
			    print(i)



	#%%
	# Dimensionality reduction/classification using LDA
	from sklearn.preprocessing import StandardScaler
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.model_selection import KFold

	roi_ID = ['v1','v2','v4','it','h','out']

	acc = np.zeros((6,5))
	for idx in range(len(roi_ID)):

	    sc = StandardScaler()
	    lda = LinearDiscriminantAnalysis()
	    kf = KFold(n_splits=5, shuffle = True)

	    X = sc.fit_transform(nBli[roi_ID[idx]])
	    y = np.zeros((800)); y[:400] = 1; i= 0
	    for train_index, test_index in kf.split(X):
	        X_train, X_test = X[train_index], X[test_index]
	        y_train, y_test = y[train_index], y[test_index]

	        X_lda = lda.fit_transform(X_train, y_train)
	        acc[idx,i] = lda.score(X_test, y_test); i+=1

	    print(idx);
	print(acc)
	avg_acc[:,r] = np.mean(acc,1)

#%%
mean = np.array([0.52, 0.53, 0.58, 0.58, 0.69, 0.71]); std = np.array([0.01, 0.01, 0.02, 0.01, 0.03, 0.02]); plt.plot(mean); plt.fill_between(range(6), mean-std, mean+std, alpha = 0.25)  # literate FR-EN bilingual
mean = np.array([0.81, 0.84, 0.82, 0.77, 0.93, 0.96]); std = np.array([0.02, 0.01, 0.01, 0.02, 0.04, 0.01]); plt.plot(mean); plt.fill_between(range(6), mean-std, mean+std, alpha = 0.25) # literate CH-EN bilingual
mean = np.array([0.5 , 0.53, 0.54, 0.53, 0.54, 0.55]); std = np.array([0.01, 0.01, 0.01, 0.02, 0.01, 0.02]); plt.plot(mean); plt.fill_between(range(6), mean-std, mean+std, alpha = 0.25) # Illiterate network
mean = np.array([0.79, 0.82, 0.81, 0.77, 0.83, 0.88]); std = np.array([0.02, 0.02, 0.02, 0.03, 0.05, 0.01]); plt.plot(mean); plt.fill_between(range(6), mean-std, mean+std, alpha = 0.25) # illiterate CH-EN bilingual

plt.plot([0,5],[.5,.5],'k--')
plt.ylabel('Classification accuracy')
plt.xticks(range(6), ['V1','V2','V4','IT','Dense','Output'])
plt.ylim([0.45,1])
plt.legend(['English-French (Literate)','English-Chinese (Literate)','English-French (Illiterate)','English-Chinese (Illiterate)'])
plt.savefig('lang_class.pdf')

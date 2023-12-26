# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:17:18 2021

@author: Aakash
"""
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn, optim
import os, sys
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
from training_code import clean_cornets
from torchvision import datasets, transforms

#%%
nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
literate = 1
if literate:
	net = clean_cornets.CORNet_Z_nonbiased_words()
	checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
# 	checkpoint = torch.load('../NN_EN_CH/models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
# 	checkpoint = torch.load('../NN_english/training_code/save/save_lit_no_bias_z_78_full_nomir.pth.tar')['state_dict']
# 	checkpoint = torch.load('../NN_chinese/save/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
# 	checkpoint = torch.load('../NN_french/models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']

else:
	net = clean_cornets.CORnet_Z_tweak()
	checkpoint = torch.load('models/save_illit_z_79_full_nomir.pth.tar')['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
net.eval()


data_dir = ['stimuli/wordselective_stimuli/']
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

    print(idx)
np.mean(acc,1).round(2)


#%% Can we improve the accuracy any further

if 1:
	from sklearn.decomposition import PCA
	pcax = PCA(n_components=80)
	idx = 5
	kf = KFold(n_splits=5, shuffle = True)
	y = np.zeros((800)); y[:400] = 1; i= 0
	X = np.array(nBli[roi_ID[idx]])

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		pcax.fit(X_train)
		pc_tr = pcax.transform(X_train)
		pc_tr = np.c_[pc_tr, np.ones((np.shape(pc_tr)[0],1))]
		tr_label = y_train
		weights = (tr_label@np.linalg.pinv(pc_tr).T).T

		tr_pred = pc_tr@weights
		mtr_pred = np.mean(tr_pred); tr_pred[tr_pred >= mtr_pred]=1;	tr_pred[tr_pred < mtr_pred]=0
		train_acc = np.mean(tr_pred == y_train)

		pc_ts = pcax.transform(X_test)
		pc_ts = np.c_[pc_ts , np.ones((np.shape(pc_ts)[0],1))]
		ts_pred = pc_ts@weights
		mts_pred = np.mean(ts_pred); ts_pred[ts_pred >= mts_pred]=1;	ts_pred[ts_pred < mts_pred]=0
		test_acc = np.mean(ts_pred == y_test)
		print(train_acc, test_acc)


if 0:
	idx = 5
	sc = StandardScaler()
	lda = LinearDiscriminantAnalysis()
	kf = KFold(n_splits=5, shuffle = True)

	# X = sc.fit_transform(nBli[roi_ID[idx]])
	X = np.array(nBli[roi_ID[idx]])

	y = np.zeros((800)); y[:400] = 1; i= 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		X_lda = lda.fit_transform(X_train, y_train)
		acc[idx,i] = lda.score(X_test, y_test); i+=1

	print(np.mean(acc[idx,:]))

#%%
# mean = [0.48, 0.49, 0.52, 0.47, 0.49, 0.5];  std = [0.03, 0.02, 0.06, 0.05, 0.02, 0.02]; plt.errorbar(range(6), mean, std) # Illiterate network
# mean = [0.5 , 0.51, 0.56, 0.55, 0.56, 0.58]; std = [0.01, 0.04, 0.02, 0.04, 0.04, 0.02]; plt.errorbar(range(6), mean, std) # literate FR-EN bilingual
# mean = [0.49, 0.51, 0.54, 0.53, 0.55, 0.59]; std = [0.03, 0.02, 0.04, 0.03, 0.05, 0.02]; plt.errorbar(range(6), mean, std) # literate CH-EN bilingual
# mean = [0.52, 0.52, 0.56, 0.51, 0.56, 0.58]; std = [0.01, 0.03, 0.03, 0.05, 0.04, 0.03]; plt.errorbar(range(6), mean, std) # literate EN
# mean = [0.51, 0.52, 0.5 , 0.54, 0.52, 0.57]; std = [0.06, 0.04, 0.04, 0.01, 0.03, 0.02]; plt.errorbar(range(6), mean, std) # literate CH
# mean = [0.5 , 0.51, 0.51, 0.56, 0.58, 0.58]; std = [0.01, 0.03, 0.02, 0.06, 0.03, 0.04]; plt.errorbar(range(6), mean, std) # literate FR

# mean = [0.48, 0.49, 0.52, 0.47, 0.49, 0.5];  std = [0.03, 0.02, 0.06, 0.05, 0.02, 0.02]; plt.plot(mean) # Illiterate network
# mean = [0.5 , 0.51, 0.56, 0.55, 0.56, 0.58]; std = [0.01, 0.04, 0.02, 0.04, 0.04, 0.02]; plt.plot(mean) # literate FR-EN bilingual
# mean = [0.49, 0.51, 0.54, 0.53, 0.55, 0.59]; std = [0.03, 0.02, 0.04, 0.03, 0.05, 0.02]; plt.plot(mean) # literate CH-EN bilingual
# mean = [0.52, 0.52, 0.56, 0.51, 0.56, 0.58]; std = [0.01, 0.03, 0.03, 0.05, 0.04, 0.03]; plt.plot(mean) # literate EN
# mean = [0.51, 0.52, 0.5 , 0.54, 0.52, 0.57]; std = [0.06, 0.04, 0.04, 0.01, 0.03, 0.02]; plt.plot(mean) # literate CH
# mean = [0.5 , 0.51, 0.51, 0.56, 0.58, 0.58]; std = [0.01, 0.03, 0.02, 0.06, 0.03, 0.04]; plt.plot(mean) # literate FR


# plt.plot([0,5],[.5,.5],'k--')
# plt.ylabel('Classification accuracy')
# plt.xticks(range(6), ['V1','V2','V4','IT','Dense','Output'])
# plt.ylim([0.45,.65])
# plt.legend(['Illiterate','FR-EN', 'CH-EN','EN', 'CH', 'FR'])
# plt.title('French vs English - 500 words')
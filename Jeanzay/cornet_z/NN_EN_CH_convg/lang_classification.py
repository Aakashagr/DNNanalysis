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
import clean_cornets
from torchvision import datasets, transforms

#%%
nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
literate = 1
if literate:
	net = clean_cornets.CORNet_Z_nonbiased_words()
	checkpoint = torch.load('save/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
# 	checkpoint = torch.load('../NN_EN_FR/models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
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


data_dir = ['../NN_EN_CH/stimuli/wordselective_stimuli/']
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

#%%

plt.plot(np.mean(acc,1))
plt.plot([0,5],[.5,.5],'k--')
plt.ylabel('Classification accuracy')
plt.xticks(range(6), ['V1','V2','V4','IT','Dense','Output'])
plt.ylim([0.45,1])
plt.title('Chinese vs English')
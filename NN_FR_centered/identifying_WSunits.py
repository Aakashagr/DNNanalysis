# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:24:08 2021

Identifying word selective units
@author: Aakash
"""
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
from torchvision import datasets, transforms
from training_code import clean_cornets

#%%
data_dir = 'stimuli/wordselective_stimuli/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80, shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['h'] = [];

net = clean_cornets.CORNet_Z_nonbiased_words()
checkpoint = torch.load('save_rep2/save_lit_no_bias_z_54_full_nomir.pth.tar',map_location=torch.device('cpu'))['state_dict']

# net = clean_cornets.CORNet_Z_biased_words()
# checkpoint = torch.load('save_rep1/save_lit_bias_z_79_full_nomir.pth.tar')['state_dict']

# net = clean_cornets.CORnet_Z_tweak()
# checkpoint = torch.load('save_rep4/save_pre_z_79_full_nomir.pth.tar')['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(10):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    _, _, _, _, varh, _ = net(stimtemp.float())
    nBli['h'].extend(varh.detach().numpy())
    print(i)

#%% Identify word selective units based on 3 standard deviations above the mean
qo = np.array(np.arange(0,400));
qf = np.array(np.arange(400,800));
data_d = np.array(nBli['h'])

neuid_f = []; neuid_e = [];
for unit in range(np.size(data_d,1)):
    Frmean = np.mean(data_d[qf,unit])
    Objmean   = np.mean(data_d[qo,unit])
    Objstdev  = np.var(data_d[qo,unit])**0.5

    if Frmean >= Objmean + 3*Objstdev:
        neuid_f.append(unit)

print(['% of French word-selective units: '+ str(np.size(neuid_f)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_f))])


# import pickle
# with open('WSunits_lit.pkl','wb') as f:
# 	pickle.dump(neuid_f, f)
#%%  Accuracy of the trained network
acc_nb = np.load('save_rep2/cat_scores_lit_no_bias_z_full_nomir.npy')
# acc_b = np.load('save_rep2/cat_scores_lit_bias_z_full_nomir.npy')
acc_ili = np.load('save_rep2/cat_scores_pre_z_full_nomir.npy')

plt.plot(np.mean(acc_nb[:,:1000],1)); plt.plot(np.mean(acc_nb[:,1000:],1))
# plt.plot(np.mean(acc_b[:,:1000],1)); plt.plot(np.mean(acc_b[:,1000:],1))
plt.plot(np.mean(acc_ili[:,:1000],1)); plt.plot(np.mean(acc_ili[:,1000:],1))

plt.ylabel('Accuracy'); plt.xlabel('Epoch'); # plt.title('Repeat 4')
# plt.legend(['Objects','French'])

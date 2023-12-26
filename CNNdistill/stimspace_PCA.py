# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:40:45 2023

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
from sklearn.decomposition import PCA
import os
import torchvision.transforms as transforms

# images = loadmat('baoimg.mat').get('img')
from clean_cornets import CORNet_Z_nonbiased_words,CORnet_Z_tweak

#%%
from torchvision import datasets

# data_dir = 'images/'
data_dir = 'stimuli/train_stimuli_all-lang/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}

wordlist = chosen_datasets['train'].classes

#%%
dataiter = iter(dataloaders['train'])

# net = CORNet_Z_nonbiased_words()
# checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
net = CORnet_Z_tweak()
checkpoint = torch.load('models/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

nBli = {};  nBli['h'] = []; 
for i in range(70):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varIt,varh, varOut = net(stimtemp.float())
    nBli['h'].extend(varIt.detach().numpy())
    # nBli['h'].extend(stimtemp.detach().numpy()[:,1,:,:].reshape(100,224*224))
    print(i)


#%% Perfoming PCA
# npc = 10
# expvar = np.zeros((7,npc))
# for i in range(7):
# 	pcax = PCA(n_components=npc)
# 	out = pcax.fit_transform(nBli['h'][i*1000:(i+1)*1000])
# 	expvar[i,:] = np.cumsum(pcax.explained_variance_ratio_)
# # 	print(np.where(expvar[i,:] > 0.9)[0][0])

# plt.plot(expvar.T)
# plt.legend(wordlist)
#%%
from scipy.spatial.distance import pdist

dmean = []
for i in range(7):
	print(i)
	dmean.append(np.mean(pdist(nBli['h'][i*1000:(i+1)*1000], metric= 'euclidean')))

plt.figure()
plt.plot(dmean)




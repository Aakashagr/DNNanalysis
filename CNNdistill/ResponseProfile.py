# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:36 2022

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
from clean_cornets import CORNet_Z_nonbiased_words

#%%
from torchvision import datasets, transforms

data_dir = 'stimuli/neuraltuning/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['out'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(1):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    _,_,_,varIt,varh, varOut = net(stimtemp.float())
    nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['out'].extend(varOut.detach().numpy())
    print(i)

#%%
wordlist = ['SSSSSSSS','XXSSSSXX','SSXXXXXX','XXXXXXSS','UUUUUUUU','XXUUUUXX','UUXXXXXX','XXXXXXUU',
            'CXXXXXXX','XCXXXXXX','GXXXXXXX','XACGIOSX','XXXXXXXE','XXXXXXEX','XXXXXXXG','XCEFGSXX']


x = np.array(nBli['h'])
out = x[:,[282,270,23,53]]

nid = [282,270,23,53]
for i in range(4):
    plt.figure(figsize = [15,4])
    plt.plot(out[:,i])
    plt.xticks(ticks = np.arange(16), labels=wordlist,rotation =45, fontsize = 16);
    plt.ylabel('Mean response')
    plt.title(nid[i])

# plt.figure()
# plt.pcolor(out,cmap = 'jet')
# plt.yticks(ticks = np.arange(16)+.5, labels=wordlist);
# plt.xticks(ticks = np.arange(4)+.5, labels=['282','270','23','53'],rotation =45);


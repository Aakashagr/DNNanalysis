# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:29:36 2021

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
import sys, os
import torchvision.transforms as transforms
import string
from  collections import defaultdict
from sklearn.linear_model import LassoCV
from torchvision import datasets, transforms
from alexnet_def_feat import AlexNet

data_dir = 'stimuli/RepSpace/'
transform = {'train': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])

#%%

net = AlexNet(num_classes = 1000)
# checkpoint = torch.load('save/save_Alexnet_lit_29.pth.tar')['state_dict']
checkpoint = torch.load('alexnet_trained.pth')
model_dict =["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight",
			 "conv3.0.bias", "conv4.0.weight", "conv4.0.bias", "conv5.0.weight", "conv5.0.bias",
			 "fc6.1.weight", "fc6.1.bias", "fc7.1.weight", "fc7.1.bias", "fc8.1.weight", "fc8.1.bias"]

state_dict={}; i=0
for k,v in checkpoint.items():
	state_dict[model_dict[i]] =  v
	i+=1

net.load_state_dict(state_dict); net.eval()
for param in net.parameters():
    param.requires_grad = False

feat = {}; feat[1] = []; feat[2] = []; feat[3] = []

for i in range(18):
    stimtemp, classes = next(dataiter)
    _,_,_,_,_,L6,L7,L8 = net(stimtemp.float())
    feat[1].extend(L6.detach().numpy())
    feat[2].extend(L7.detach().numpy())
    feat[3].extend(L8.detach().numpy())
    print(i)

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pcdata = np.array(feat[1])
x_train = pcdata[:1224,:]
x_test = pcdata[1224:,:]
# x_train = pcdata[1224:-80,:]
# x_test = pcdata[1224:,:]


x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

pcax = PCA(n_components= 10 ,whiten = 0)
pcax.fit(x_train)

x_train = pcax.transform(x_train)
x_test = pcax.transform(x_test)

#%% Plotting the PC space
# plt.scatter(x_train[:,0],x_train[:,1])
color = ['m','c','g','orange']
for i in range(4):
	idx = np.arange(i*20,(i+1)*20)
	plt.scatter(x_test[idx,0],x_test[idx,1],c = color[i])
# 	plt.scatter(x_train[idx,0],x_train[idx,1])
plt.scatter(x_test[-60:-40,0],x_test[-60:-40,1],c = 'k')

plt.legend(['Stubby','Faces','Bodies','NML','words'])
plt.title('Illiterate - Alexnet, FC8')


#%%

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
color = ['m','c','g','orange']
for i in range(4):
	idx = np.arange(i*20,(i+1)*20)
	ax.scatter3D(x_test[idx,0],x_test[idx,1],x_test[idx,2],c = color[i])
ax.scatter3D(x_test[-60:-40,0],x_test[-60:-40,1],x_test[-60:-40,2],c = 'k')


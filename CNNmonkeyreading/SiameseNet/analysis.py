# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:22:46 2023

@author: Aakash
"""

import numpy as np
import matplotlib.pyplot as plt
import clean_cornets
from torch import nn
import torch
plt.rcParams['pdf.fonttype'] = 42

import os
import torchvision.transforms as transforms
from torchvision import datasets
from scipy.stats import spearmanr, pearsonr

#%%
x = np.load('models/training_loss.npz')
plt.plot([0.5375,
 0.6569444444444444,
 0.7107638888888889,
 0.7680555555555556,
 0.7670138888888889,
 0.8083333333333333,
 0.8378472222222222,
 0.9149305555555556,
 0.9041666666666667,
 0.9184027777777778,
 0.9315972222222222,
 0.9440972222222223,
 0.9465277777777777,
 0.9479166666666666,
 0.9378472222222223,
 0.940625,
 0.93125,
 0.9361111111111111,
 0.9135416666666667,
 0.9131944444444444])
plt.plot([0.8055555555555556,
 0.6722222222222223,
 0.8833333333333333,
 0.6555555555555556,
 0.7555555555555555,
 0.8972222222222223,
 0.9305555555555556,
 0.8333333333333334,
 0.9166666666666666,
 0.8777777777777778,
 0.925,
 0.9305555555555556,
 0.9361111111111111,
 0.9222222222222223,
 0.9166666666666666,
 0.9138888888888889,
 0.9194444444444444,
 0.9166666666666666,
 0.8805555555555555,
 0.9055555555555556])

plt.legend(['Training accuracy','Val accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0.5,1])
#%%

data_dir = 'stimuli/neural_stim/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 500,shuffle = False) for x in ['train']}
    
#%%

net = clean_cornets.CORnet_Z_tweak()
net = nn.Sequential(*list(net.children())[:-4])
net.load_state_dict(torch.load('models/cornet_lit_epoch2.pth', map_location = 'cpu'))

lit = {};  lit['v1'] = [];  lit['v2'] = [];  lit['v4'] = []; lit['it'] = []; lit['h'] = [];
dataiter = iter(dataloaders['train'])
stimtemp, classes = next(dataiter)
# lit['v1'], lit['v2'], lit['v4'], lit['it'], lit['h'] = net(stimtemp.float())
lit['h'] = net(stimtemp.float())

#%%
resp = lit['h'].detach().numpy()

dneu = np.zeros((36,8))
for w in range(36):
	for nw in range(8):
		r = pearsonr(resp[w*5+2], resp[180+w*8+nw])
		dneu[w,nw] = r[0]

plt.figure()
plt.bar(range(8), np.mean(dneu,0))
plt.xticks(np.arange(8)-0.5, labels = ['all different letter',	'consonant/vowel string',	
							   'NW - diff. structure',	'W - diff. structure',	
							   'NW - same structure',	'W - same structure',	
							   'Transposition', 'Substitution'], rotation = 45)
plt.ylabel('Mean Similarity to word representation')
plt.title('Literate network')
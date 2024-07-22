# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:48:20 2023

The word selective units for which the responses needs to be generated should be 
uncommented manually and also the filename to save the figure. 

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle

#%% Loading relevant word selective units

layer = 'v2'
with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)

# Excitatory units
# wordSelUnit = [39316,39317,39318,14228,14229,14230,45186,45187,45188,18530,18531,18532]
# wordSelUnit = [1164,467,11351] # IT units for v4-IT connection
# wordSelUnit = [84290,5110,7487,89811] # V1 units
# wordSelUnit = [4219,1671] # V4 unit for v2-v4 connection

# Inhibitory units
# wordSelUnit = [21284,21285,21286,5800,5801,5802, 32446,32447,32448,46754,46755,46768]

# Mid position units all
# wordSelUnit = [6180,6181,6182,21273,21274,21275, 45186,45187,45188,1658,1659,1660]

# V2-V4 connectivity units
wordSelUnit = [63157,63178,63179,78018,78045,78046, 8214,8215,8216,51390,51391,51392]
# wordSelUnit = [87461,87482,87483,8214,8215,8216, 78018,78045,78046,41198,41199,41227]


#%% Extracting CNN activations

data_dir = 'stimuli/PosTuning_letters/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['v4'] = []; nBli['v2'] = []
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
	if 'module.' in key:
		checkpoint[key.replace('module.', '')] = checkpoint[key]
		del checkpoint[key]
net.load_state_dict(checkpoint)
# net.eval()

for i in range(2):
	stimtemp, classes = next(dataiter)
	varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
	nBli['v2'].extend(varv2.detach().numpy())
	nBli['v4'].extend(varv4.detach().numpy())
	nBli['h'].extend(varh.detach().numpy())
	nBli['it'].extend(varit.detach().numpy())
	
#%% Plotting the data
	
fig, axs = plt.subplots(2,6, figsize=(30,20), facecolor='w', edgecolor='k')
axs = axs.ravel();	
for i in range(len(wordSelUnit)):

	# Visualizing the coefficients
	out = np.array(nBli[layer])[:,wordSelUnit[i]]
	charcoef = np.reshape(out,[26,8])
	maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
	for r in range(np.size(charcoef,0)):
#		 strchar = string.ascii_lowercase[r]
		strchar = string.ascii_uppercase[r]
		for c in range(np.size(charcoef,1)):
			strcol = 'red' if charcoef[r,c] >0 else 'blue'
			axs[i].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
			axs[i].set_xticks(np.arange(0.5,9,1)); 
			axs[i].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
			axs[i].set_yticks(np.arange(0.5,27,1));
			axs[i].set_yticklabels([]);
			axs[i].yaxis.set_ticks_position('none')


	axs[i].set_title('unit #: ' + str(wordSelUnit[i]), fontsize = 16)
	
# fig.savefig('plots/connectivity/letter_tuning_'+layer+'_rf_midpos.pdf')

fig.savefig('plots/connectivity/letter_tuning_v2v4conn_layer_'+layer+'_rf_1.pdf')
	

  
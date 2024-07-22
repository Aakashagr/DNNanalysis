# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:38:11 2024

@author: Aakash
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
import string
from sklearn.linear_model import LassoCV
from torchvision import datasets
from clean_cornets import CORNet_Z_nonbiased_words
import pickle

#%%

data_dir = 'stimuli/wordsets_1000cat_8ex/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}
wordlist = chosen_datasets['train'].classes

#%%
dataiter = iter(dataloaders['train'])

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)

layer_idx = ['v1','v2','v4','it','h']

nBli = {}; 
for l in layer_idx:
	nBli[l] = []; 
	
for i in range(100):
    stimtemp, classes = next(dataiter)
    varv1,varv2,varv4,varIt,varh, _ = net(stimtemp.float())
    nBli['v1'].extend(varv1.detach().numpy())
    nBli['v2'].extend(varv1.detach().numpy())
    nBli['v4'].extend(varv1.detach().numpy())
    nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    print(i)

#%% Model fitting on words 
	
strlist = list(string.ascii_lowercase) # storing characters from A-Z
stimword = np.transpose(np.tile(wordlist,(8, 1))).flatten()

##### Building the regression matrix
Xmat = np.zeros((len(stimword), 26*8))
for i, seq in enumerate(stimword):
    mpt = round(len(seq)/2)
    for j, char in enumerate(seq[:mpt]):  # Left half of the words
        if char in strlist:
            pid = (strlist.index(char)*8) + j
            Xmat[i,pid] += 1

    for j, char in enumerate(seq[mpt:][::-1]):  # right half of the words
        if char in strlist:
            pid = (strlist.index(char)*8) + 7 - j
            Xmat[i,pid] += 1
	

# Initializing variables
coefMat = {};  rfit = {};
for l in layer_idx:
	coefMat[l] = [];
	rfit[l] = [];
	
# Fitting model across all layers
for layer in layer_idx:
	with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
	 	wordSelUnit = pickle.load(f)
		 
	x = np.array(nBli[layer]) #stimulus response
	out = x[:,wordSelUnit]  # Analysing properties of individual word selective units

	##### Initializing variables
	qrfit = np.zeros(len(wordSelUnit))
	qcoefMat = np.zeros((len(wordSelUnit), 26*8))
	
	##################################################### Model fitting
	for npc in np.arange(len(wordSelUnit)):
	    if np.var(out[:,npc]) < 0.01:
	        print('ignored id: ' + str(npc))
	    else:             
	        reg = LassoCV(cv=5, random_state=0,max_iter=10000, n_jobs = -1).fit(Xmat, out[:,npc])
	        corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
	        qrfit[npc] = corrval**2
	        qcoefMat[npc,:] = reg.coef_
	        print(npc, qrfit[npc])
	
	rfit[layer] = qrfit
	coefMat[layer] = qcoefMat
	
#%%
layer = 'v4'
plt.hist(rfit[layer])
plt.xlabel('Model fit score R2, '+ layer)
plt.ylabel('# of units')

#%%

if 1:

    fig, axs = plt.subplots(1,4, figsize=(15,10), facecolor='w', edgecolor='k')
    axs = axs.ravel();
    for i,val in enumerate([243,244,709,710]):
        print(val)
        # Visualizing the coefficients
        charcoef = np.reshape(coefMat[layer][val,:],[26,8])
        maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
        for r in range(np.size(charcoef,0)):
    #         strchar = string.ascii_lowercase[r]
            strchar = string.ascii_uppercase[r]
            for c in range(np.size(charcoef,1)):
                strcol = 'red' if charcoef[r,c] >0 else 'blue'
                axs[i].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
                axs[i].set_xticks(np.arange(0.5,9,1)); axs[i].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
                axs[i].set_yticks(np.arange(0.5,27,1)); axs[i].set_yticklabels([]);
                axs[i].yaxis.set_ticks_position('none')
    
    
        axs[i].set_title('unit #: ' + str(wordSelUnit[val])+ ': r2 = '+str(round(rfit[layer][val],2)), fontsize = 16)
    
	
#%%
nws = np.shape(coefMat[layer])[0]
meancoef = np.zeros((nws,8))
for i in range(nws):
    qtemp = np.mean(np.reshape(coefMat[layer][i,:],[26,8]),0)
    meancoef[i,:] = qtemp/max(qtemp)

maxid = np.argmax(meancoef, axis = 1)

for i in range(8):
	qid = np.where(maxid == i)[0]
	plt.plot(np.mean(meancoef[qid,:],0))

plt.xticks(ticks =range(8),labels = [str(i+1) for i in range(8)])
plt.xlabel('Letter Position')
plt.ylabel('mean Normalized coefficient')
plt.savefig('plots/normcoeff_'+layer+'.pdf')
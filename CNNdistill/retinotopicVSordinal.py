# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:22:54 2023

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
from sklearn.linear_model import LassoCV, RidgeCV
from torchvision import datasets
from clean_cornets import CORNet_Z_nonbiased_words

#%%

data_dir = 'stimuli/wordsets_1000cat/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}

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
net.eval()

nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; 
for i in range(10):
	stimtemp, classes = next(dataiter)
	# nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
	varv1,varv2,varv4,varIt,varh, _ = net(stimtemp.float())
	nBli['v1'].extend(varv1.detach().numpy())
	nBli['v2'].extend(varv2.detach().numpy())
	nBli['v4'].extend(varv4.detach().numpy())
	nBli['it'].extend(varIt.detach().numpy())
	nBli['h'].extend(varh.detach().numpy())
	print(i)

#%% Model fitting on words and testing on Vinckier stimuli
import pickle
rfitO = {}; coefMatO = {}
rfitR = {}; coefMatR = {}

for layer in list(nBli.keys())[-1:]:
	print('Layer: ' + layer)
	with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
	 	wordSelUnit = pickle.load(f)
	
	############### Loading 1000 word stimuli
	x = np.array(nBli[layer]) #stimulus response
	strlist = list(string.ascii_lowercase) # storing charaacters from A-Z
	stimword = wordlist
	
	# Setting up the parameters to perform PC
	out = x[:,wordSelUnit]  # Analysing properties of individual word selective units
	
	##### Building the regression matrix
	XmatO = np.zeros((len(stimword), 26*8))
	for i, seq in enumerate(stimword):
		mpt = round(len(seq)/2)
		for j, char in enumerate(seq[:mpt]):  # Left half of the words
			if char in strlist:
				pid = (strlist.index(char)*8) + j
				XmatO[i,pid] += 1
	
		for j, char in enumerate(seq[mpt:][::-1]):  # right half of the words
			if char in strlist:
				pid = (strlist.index(char)*8) + 7 - j
				XmatO[i,pid] += 1
	
	XmatR = np.zeros((len(stimword), 26*8)) 
	for i, seq in enumerate(stimword):
		offset = int(4-len(seq)/2)
		for j, char in enumerate(seq):  
			if char in strlist:
				pid = (strlist.index(char)*8) + j + offset
				XmatR[i,pid] += 1 
	
	##### Initializing variables
	rfitO[layer] = np.zeros(len(wordSelUnit)); rfitO[layer][:] = np.nan
	rfitR[layer] = np.zeros(len(wordSelUnit)); rfitR[layer][:] = np.nan

	coefMatO[layer] = np.zeros((len(wordSelUnit), 26*8))
	coefMatR[layer] = np.zeros((len(wordSelUnit), 26*8))	
	##################################################### Model fitting
	for npc in np.arange(len(wordSelUnit)):
		if np.var(out[:,npc]) > 0.01: 
			reg = LassoCV(cv=3,max_iter=10000, n_jobs = -1).fit(XmatO, out[:,npc])
# 			reg = RidgeCV(cv=3).fit(XmatO, out[:,npc])
			corrval,pval = scipy.stats.pearsonr(XmatO@reg.coef_, out[:,npc])
			rfitO[layer][npc] = corrval**2
			coefMatO[layer][npc,:] = reg.coef_
			
			reg = LassoCV(cv=3 ,max_iter=10000, n_jobs = -1).fit(XmatR, out[:,npc])
# 			reg = RidgeCV(cv=3).fit(XmatR, out[:,npc])
			corrval,pval = scipy.stats.pearsonr(XmatR@reg.coef_, out[:,npc])
			rfitR[layer][npc] = corrval**2
			coefMatR[layer][npc,:] = reg.coef_		
		
	print(np.nanmean(rfitR[layer]), np.nanmean(rfitO[layer]))

#%% 
from scipy.stats import ttest_rel
for layer in nBli.keys():
	print(ttest_rel(rfitR[layer],rfitO[layer], nan_policy='omit'))

#%%
meanfit = np.zeros((5,2))
stdfit = np.zeros((5,2))
for i,layer in enumerate(nBli.keys()):
	meanfit[i,0] = np.nanmean(rfitR[layer]); 	meanfit[i,1] = np.nanmean(rfitO[layer])
	stdfit[i,0] = np.nanstd(rfitR[layer]); 	stdfit[i,1] = np.nanstd(rfitO[layer])


plt.figure()
plt.bar(np.arange(5)-.2, meanfit[:,0], width = 0.3)
plt.bar(np.arange(5)+.2, meanfit[:,1], width = 0.3)
plt.errorbar(np.arange(5)-.2, meanfit[:,0],  stdfit[:,0],color ='k')
plt.errorbar(np.arange(5)+.2, meanfit[:,1],  stdfit[:,1],color ='k')
plt.xticks(range(5), nBli.keys())
plt.ylabel('Correlation coefficient')
plt.xlabel('Layers')
plt.legend(['Retinotopic','Ordinal'])
# plt.savefig('modelfit.pdf', dpi = 300)

#%%
nws = np.shape(coefMatO[layer])[0]
meancoef = np.zeros((nws,8))
for i in range(nws):
	 qtemp = np.nanmean(np.reshape(coefMatO[layer][i,:],[26,8]),0)
	 meancoef[i,:] = qtemp/max(qtemp)

maxid = np.argmax(meancoef, axis = 1)

for i in range(8):
 	qid = np.where(maxid == i)[0]
 	plt.plot(np.nanmean(meancoef[qid,:],0))

plt.xticks(ticks =range(8),labels = [str(i+1) for i in range(8)])
plt.xlabel('Letter Position')
plt.ylabel('mean Normalized coefficient')
plt.savefig('avgCoef.pdf', dpi = 300)


#%%

sortid = np.argsort(-rfitO[layer]) # Sorting in descending order
visid = sortid[:14]

# Initializing the variables
max_len = max(map(len, stimword)); 
fig, axs = plt.subplots(2,7, figsize=(25,20), facecolor='w', edgecolor='k')
axs = axs.ravel();
strlist = list(string.ascii_lowercase) # storing charaacters from A-Z


for i,val in enumerate(visid):
    print(val)
    # Visualizing the coefficients
    charcoef = np.reshape(coefMatO[layer][val,:],[26,8])
    maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
    for r in range(np.size(charcoef,0)):
        strchar = string.ascii_uppercase[r]
        for c in range(np.size(charcoef,1)):
            strcol = 'red' if charcoef[r,c] >0 else 'blue'
            axs[i].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
            axs[i].set_xticks(np.arange(1,9,1)); axs[i].set_xticklabels(['1','2','3','4','5','6','7','8']);
            axs[i].set_yticks(np.arange(0.5,27,1)); axs[i].set_yticklabels([]);
            axs[i].yaxis.set_ticks_position('none') 


    axs[i].set_title('unit #: ' + str(wordSelUnit[val])+ ': r = '+str(round(rfitO[layer][val],2)))

plt.savefig("manuscript_codes/letterRF.pdf")

"""
Created on Fri Sep  8 11:41:49 2023

@author: Aakash
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
import string
from sklearn.linear_model import LassoCV

from clean_cornets import CORNet_Z_nonbiased_words

#%%
from torchvision import datasets

# data_dir = 'images/'
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

#%%
import pickle
with open('WSunits/rep0/WSunits_lit_h.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)
	 
# Comparing both ends models
stimword = wordlist
x = np.array(nBli['h']) #stimulus 
out = x[:,wordSelUnit]  # Analysing properties of individual word selective units
strlist = list(string.ascii_lowercase) # storing charaacters from A-Z

# Building the regression matrix for Edge-aligned
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

aic_nol = np.zeros((len(wordSelUnit)))
rfit_nol = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
	yobs = out[:,npc]	   
	reg = LassoCV(cv=5, random_state=0,max_iter=10000,n_jobs =-1).fit(Xmat, yobs)   
	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
	rfit_nol[npc] = corrval
	sse = np.sum((Xmat@reg.coef_ - out[:,npc])**2)
	nsam = np.size(out,0); npars = len(reg.coef_)
	aic_nol[npc] = nsam*np.log(sse/nsam) + (2*npars) + (2*npars*(npars+1)/(nsam-npars-1))
	
print('edge-aligned model done')  


# Fitting retinotopic models (word centred model)
Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
	offset = int(4-len(seq)/2)
	for j, char in enumerate(seq):
		if char in strlist:
			pid = (strlist.index(char)*8) + j + offset
			Xmat[i,pid] += 1 

rfit_ret = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
	yobs = out[:,npc]	   
	reg = LassoCV(cv=5, random_state=0,max_iter=10000,n_jobs =-1).fit(Xmat, yobs)   
	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
	rfit_ret[npc] = corrval
 
print('word-centred model done')  

# Comparing left-right vs right-left model fits

Xmat = np.zeros((len(stimword), 26*8)) 
for i, seq in enumerate(stimword):
	for j, char in enumerate(seq): 
		if char in strlist:
			pid = (strlist.index(char)*8) + j
			Xmat[i,pid] += 1 

rfit_lr = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
	yobs = out[:,npc]	   
	reg = LassoCV(cv=5, random_state=0,max_iter=10000).fit(Xmat, yobs)   
	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
	rfit_lr[npc] = corrval
	
print('Left aligned model done') 
 
###########  Filling the location from right to left
# Xmat = np.zeros((len(stimword), 26*8)) 
# for i, seq in enumerate(stimword):
# 	pos_offset = 8 - len(seq)
# 	for j, char in enumerate(seq):
# 		if char in strlist:
# 			pid = (strlist.index(char)*8) + j + pos_offset
# 			Xmat[i,pid] += 1 


# rfit_rl = np.zeros((len(wordSelUnit)))
# for npc in range(len(wordSelUnit)): 
# 	yobs = out[:,npc]	   
# 	reg = LassoCV(cv=5, random_state=0,max_iter=10000).fit(Xmat, yobs)   
# 	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
# 	rfit_rl[npc] = corrval

# print('Right aligned model done') 


#%% Both edge models
# Xmat = np.zeros((len(stimword), 26*8*2)) 
# for i, seq in enumerate(stimword):
# 	for j, char in enumerate(seq):
# 		if char in strlist:
# 			pid = (strlist.index(char)*16) + j
# 			Xmat[i,pid] += 1 
# 			pid = (strlist.index(char)*16) +len(seq)-j+8
# 			Xmat[i,pid] += 1 
# 		

# aic_be = np.zeros((len(wordSelUnit)))
# rfit_be = np.zeros((len(wordSelUnit)))
# for npc in range(len(wordSelUnit)): 
# 	yobs = out[:,npc]	   
# 	reg = LassoCV(cv=5, random_state=0,max_iter=10000,n_jobs =-1).fit(Xmat, yobs)   
# 	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
# 	rfit_be[npc] = corrval
# 	sse = np.sum((Xmat@reg.coef_ - out[:,npc])**2)
# 	nsam = np.size(out,0); npars = len(reg.coef_)
# 	aic_be[npc] = nsam*np.log(sse/nsam) + (2*npars) + (2*npars*(npars+1)/(nsam-npars-1))
# 	
#%% omparing both ends models with gradation

if 0:
	Xmat = np.zeros((len(stimword), 26*8)) 
	for i, seq in enumerate(stimword):
		mpt = round(len(seq)/2)
		for j, char in enumerate(seq[:mpt]):  # Left half of the words 
			if char in strlist:
				pid = (strlist.index(char)*8) + j
				Xmat[i,pid] += 1 
				if pid > strlist.index(char)*8:
					Xmat[i,pid-1] += 0.5 
				if pid < strlist.index(char)*8 + 7:
					Xmat[i,pid+1] += 0.5 
				
		for j, char in enumerate(seq[mpt:][::-1]):  # right half of the words 
			if char in strlist:
				pid = (strlist.index(char)*8) + 7 - j
				Xmat[i,pid] += 1 
				if pid > strlist.index(char)*8:
					Xmat[i,pid-1] += 0.5 
				if pid < strlist.index(char)*8 + 7:
					Xmat[i,pid+1] += 0.5
					
	
	aic_nolg = np.zeros((len(wordSelUnit)))
	rfit_nolg = np.zeros((len(wordSelUnit)))
	for npc in range(len(wordSelUnit)): 
		yobs = out[:,npc]	   
		reg = LassoCV(cv=5, random_state=0,max_iter=10000,n_jobs =-1).fit(Xmat, yobs)   
		corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
		rfit_nolg[npc] = corrval
		sse = np.sum((Xmat@reg.coef_ - out[:,npc])**2)
		nsam = np.size(out,0); npars = len(reg.coef_)
		aic_nolg[npc] = nsam*np.log(sse/nsam) + (2*npars) + (2*npars*(npars+1)/(nsam-npars-1))
		
	print('Graded edge-aligned model done')  

#%% Bigram model
bigrams = []
for lword in wordlist:
    for i in range(len(lword)-1):
        bigrams.append(lword[i:i+2])
        
bigramlist =  np.unique(bigrams); bigramlist = list(bigramlist[3:])  

Xmat = np.zeros((len(stimword), len(bigramlist))) 
for i, seq in enumerate(stimword):
    for j in range(len(seq)-1): 
        char = seq.lower()[j:j+2]
        if char in bigramlist:
            pid = (bigramlist.index(char)) 
            Xmat[i,pid] += 1 

rfit_bi = np.zeros((len(wordSelUnit)))
for npc in range(len(wordSelUnit)): 
	yobs = out[:,npc]	   
	reg = LassoCV(cv=5, random_state=0,max_iter=10000).fit(Xmat, yobs)   
	corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
	rfit_bi[npc] = corrval
	
#%%
plt.figure()
plt.scatter(rfit_ret, rfit_nol); plt.plot([0.2,1],[0.2,1])
plt.scatter(rfit_lr, rfit_nol); plt.plot([0.2,1],[0.2,1])
plt.ylabel('Edge Aligned model')
plt.xlabel('Word Centred model')
plt.axis('square')
# plt.savefig('manuscript_codes/figure1_1.pdf')

plt.figure()
plt.scatter(rfit_bi, rfit_nol); plt.plot([0.2,1],[0.2,1])
plt.ylabel('Edge Aligned model')
plt.xlabel('Bigram model')
plt.axis('square')
plt.savefig('manuscript_codes/figure1_2.pdf')
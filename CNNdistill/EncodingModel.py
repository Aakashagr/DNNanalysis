# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:18:41 2022

@author: Aakash
"""
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

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
from sklearn.linear_model import LassoCV, RidgeCV
from scipy import stats
from torchvision import datasets, transforms
from clean_cornets import CORNet_Z_nonbiased_words

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
# net.eval()

nBli = {}; nBli['it'] = []; nBli['h'] = []; nBli['v4'] = []; nBli['v1'] = []
for i in range(100):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varIt,varh, _ = net(stimtemp.float())
    # nBli['v1'].extend(varv1.detach().numpy())
    # nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    print(i)

#%% Model fitting on words and testing on Vinckier stimuli
########  Units selective for words identified by Thomas
import pickle
layer = 'h'

with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)

############### Loading 1000 word stimuli
x = np.array(nBli['h']) #stimulus response
strlist = list(string.ascii_lowercase) # storing charaacters from A-Z
stimword = np.transpose(np.tile(wordlist,(8, 1))).flatten()

# Setting up the parameters to perform PC
out = x[:,wordSelUnit]  # Analysing properties of individual word selective units

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


##### Initializing variables
rfit = np.zeros(len(wordSelUnit))
coefMat = np.zeros((len(wordSelUnit), 26*8))

##################################################### Model fitting
for npc in np.arange(len(wordSelUnit)):
    if np.var(out[:,npc]) < 0.01:
        print('ignored id: ' + str(npc))
    else:             
        reg = LassoCV(cv=5, random_state=0,max_iter=10000, n_jobs = -1).fit(Xmat, out[:,npc])
        # reg = RidgeCV(cv=5).fit(Xmat, out[:,npc])
        corrval,pval = scipy.stats.pearsonr(Xmat@reg.coef_, out[:,npc])
        rfit[npc] = corrval**2
        coefMat[npc,:] = reg.coef_
        print(npc, rfit[npc])

#%%
plt.hist(rfit)
plt.xlabel('Model fit score R2, v1')
plt.ylabel('# of units')

#%%
nws = np.shape(coefMat)[0]
meancoef = np.zeros((nws,8))
for i in range(nws):
    qtemp = np.mean(np.reshape(coefMat[i,:],[26,8]),0)
    meancoef[i,:] = qtemp/max(qtemp)

maxid = np.argmax(meancoef, axis = 1)

for i in range(8):
	qid = np.where(maxid == i)[0]
	plt.plot(np.mean(meancoef[qid,:],0))

plt.xticks(ticks =range(8),labels = [str(i+1) for i in range(8)])
plt.xlabel('Letter Position')
plt.ylabel('mean Normalized coefficient')

# plt.savefig('normcoeff.pdf')

#%%
nws = np.shape(coefMat)[0]
meancoeff = np.zeros((nws,26))
for i in range(nws):
    meancoeff[i,:] = np.mean(np.reshape(coefMat[i,:],[26,8]),1)
    
import pickle
with open('mean_coeff_h.pkl', 'wb') as f:
 	pickle.dump([meancoeff, rfit], f)
#%% plotting the receptive field
os.makedirs('plots/model_coef_'+layer+'/', exist_ok= True)

if 1:
    # Initializing the variables
	for bias in np.arange(0,len(wordSelUnit),10):

	    fig, axs = plt.subplots(1,10, figsize=(40,10), facecolor='w', edgecolor='k')
	    axs = axs.ravel();
	    for i,val in enumerate(np.arange(bias,10+bias)):
	        print(val)
	        # Visualizing the coefficients
	        charcoef = np.reshape(coefMat[val,:],[26,8])
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
	    
	    
	        axs[i].set_title('unit #: ' + str(wordSelUnit[val])+ ': r2 = '+str(round(rfit[val],2)), fontsize = 16)
	    
	    fig.savefig('plots/model_coef_'+layer+'/' +str(bias) + '.png')
	    plt.close(fig)
	
	
#%% Match to behaviour dissimilarity
from scipy.io import loadmat
from scipy.spatial.distance import pdist
dis_u = loadmat('dis_uletter.mat')['dis_uletter']
dis_l = loadmat('dis_lletter.mat')['dis_lletter']
dis_beh = (dis_u + dis_l)/2


avgcoef = []
for i in range(len(wordSelUnit)):
    avgcoef.append(np.mean(np.reshape(coefMat[i,:],[26,8]),1))
    # avgcoef.append(np.reshape(coefMat[i,:],[26,8])[:,7])


avgcoef = np.array(avgcoef).T
neudis = pdist(avgcoef, metric = 'cosine')
               
plt.figure()
plt.scatter(dis_beh, neudis)
r,p = stats.spearmanr(dis_beh, neudis, nan_policy = 'omit')
plt.title('Corr. coef = ' + str(r.round(2))+ ', p = '+ str(p.round(3)))
plt.ylabel('Estimated dissimilarity'); plt.xlabel('Behaviour dissimilarity')


#%% Match to behaviour for each position
rval = []
for Pid in range(8):
    avgcoef = []
    for i in range(len(wordSelUnit)):
        avgcoef.append(np.reshape(coefMat[i,:],[26,8])[:,Pid])
    
    avgcoef = np.array(avgcoef).T
    neudis = pdist(avgcoef, metric = 'cosine')                   
    r,p = stats.spearmanr(dis_beh, neudis, nan_policy = 'omit')
    rval.append(r)


plt.figure()
plt.bar(range(8),rval)
plt.xticks(range(8), labels = ['1','2','3','4','5','6','7','8'])
plt.ylabel('Correlation Coefficient')
plt.xlabel('Position')
plt.title('Match to behaviour letter dissimilarity')

#%% Match to behaviour for each unit
rval = []
for i in range(len(wordSelUnit)):
    avgcoef = np.zeros((26,2))
    avgcoef[:,0] = (np.mean(np.reshape(coefMat[i,:],[26,8]),1))

    neudis = pdist(avgcoef, metric = 'euclidean')                   
    r,p = stats.spearmanr(dis_beh, neudis)
    rval.append(r)
    
plt.figure()
plt.hist(rval)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Counts')
plt.title('Match to behaviour for each unit')


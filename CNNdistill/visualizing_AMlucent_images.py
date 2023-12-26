# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:10:27 2022

@author: Aakash
"""   
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

with open('WSunits/WSunits_lit_h.pkl', 'rb') as f:
     	wordSelUnit = pickle.load(f)

#%%
qdir = 'plots/Activation_maximization_lucent/rep0/literate/AM_IT/'
files = os.listdir(qdir)

imgs = []
for w in wordSelUnit:
	imgs.append(plt.imread(qdir + files[w]))
	
#%%

for qx in range(9):
	fig, axs = plt.subplots(4,2, figsize=(20,25), facecolor='w', edgecolor='k')
	axs = axs.ravel();
	for i,j in enumerate(np.arange(qx*8,(qx+1)*8)):
		axs[i].imshow(imgs[j])
		axs[i].set_title('Unit #: '+ str(wordSelUnit[j]),  fontsize = 50)
		axs[i].set_xticks([])
		axs[i].set_yticks([])
		
#%%
with open('WSunits/WSunits_lit_v4.pkl', 'rb') as f:
     	wordSelUnit_v4 = pickle.load(f)
	 
wordSelUnit_v4 = np.floor(np.array(wordSelUnit_v4)/196)
np.unique(wordSelUnit_v4)

# qdir = 'plots/Activation_maximization_lucent/rep0/literate/AM_IT/'
# files = os.listdir(qdir)

# imgs = []
# for w in wordSelUnit:
# 	imgs.append(plt.imread(qdir + files[w]))
# 	
	
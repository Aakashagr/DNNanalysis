# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:17:18 2021

@author: Aakash
"""
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import pickle
#%%
roi_ID = ['v1','v2','v4','it','h','out']
acc = np.zeros((2,5,5,5)) # lang x reps x 5 layers x 5 fold

for lid,lang in enumerate(['illit','fr']):
	if lang != 'illit':
		net = CORNet_Z_nonbiased_words()
	else:
		net = CORnet_Z_tweak()
		
	# Loading model
	for rep in range(5):
		checkpoint = torch.load('model_all-lang/save_lit_'+lang+'_rep'+str(rep)+'.pth.tar',map_location ='cpu')['state_dict']

		for key in list(checkpoint.keys()):
		    if 'module.' in key:
		        checkpoint[key.replace('module.', '')] = checkpoint[key]
		        del checkpoint[key]
		net.load_state_dict(checkpoint)
		net.eval()
		
		# Loading dataset
		data_dir = 'stimuli/EN_FR_classification/'
		nBli = {}; 
		for i in roi_ID:
			nBli[i] = []
			
		transform = {'train': transforms.Compose([transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
	
		chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
		dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 100,shuffle = False) for x in ['train']}
		dataiter = iter(dataloaders['train'])
		for i in range(20):
			stimtemp, classes = next(dataiter)
			varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
			nBli['v1'].extend(varV1.detach().numpy())
			nBli['v2'].extend(varV2.detach().numpy())
			nBli['v4'].extend(varV4.detach().numpy())
			nBli['it'].extend(varIt.detach().numpy())
			nBli['h'].extend(varh.detach().numpy())
			nBli['out'].extend(varOut.detach().numpy())
		
		
		# Training model across all layers
		for idx in range(len(roi_ID)-1):
			
			# Using only word selective untis
			with open('WSunits/rep'+str(rep)+'/WSunits_lit_'+roi_ID[idx]+'.pkl', 'rb') as f:
				wordSelUnit = pickle.load(f)
		    
			sc = StandardScaler()
			lda = LinearDiscriminantAnalysis()
			kf = KFold(n_splits=5, shuffle = True)
		
			X = sc.fit_transform(nBli[roi_ID[idx]])
			y = np.zeros((1918)); y[:959] = 1; i= 0
			X = X[:,wordSelUnit]
			for train_index, test_index in kf.split(X):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
		
				X_lda = lda.fit_transform(X_train, y_train)
				acc[lid,rep,idx,i] = lda.score(X_test, y_test); i+=1
		
		print('Repeat:'+ str(rep))

#%%
mean = np.mean(np.mean(acc[0,:,:,:],2),0); std = np.std(np.mean(acc[0,:,:,:],2),0)/np.sqrt(5);
plt.plot(mean); plt.fill_between(range(5), mean-std, mean+std, alpha = 0.25)  # Illiterate 
mean = np.mean(np.mean(acc[1,:,:,:],2),0); std = np.std(np.mean(acc[1,:,:,:],2),0)/np.sqrt(5);
plt.plot(mean); plt.fill_between(range(5), mean-std, mean+std, alpha = 0.25) # literate FR-EN bilingual

plt.plot([0,4],[.5,.5],'k--')
plt.ylabel('Classification accuracy')
plt.xticks(range(5), ['V1','V2','V4','IT','avgIT'])
plt.ylim([0.49,0.66])
plt.legend(['Illiterate n/w','Literate n/w'])
plt.savefig('plots/lang_class.pdf')

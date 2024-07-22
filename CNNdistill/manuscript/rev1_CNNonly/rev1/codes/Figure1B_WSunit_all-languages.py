# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:56:38 2022

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets
import clean_cornets

#%%
data_dir = 'stimuli/wordselective_stimuli_all-lang/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
			  transforms.ToTensor(),
			  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200, shuffle = False) for x in ['train']}

#%%


qo = np.array(np.arange(0,400)); # Object stim 
Lidx_start = np.array(np.arange(400,2400,400)); # Language stim

allmodels = ['fr','en','ch','tel','mal','en-ch','en-fr','illit']
nWS = np.zeros((len(Lidx_start),len(allmodels),5))
nWS_unique = np.zeros((len(allmodels),5))


for Mid,lang in enumerate(allmodels):
	for rep in range(5):
		dataiter = iter(dataloaders['train'])
		print(lang)
		if lang == 'illit':
			net = clean_cornets.CORnet_Z_tweak()
		else:
			net = clean_cornets.CORNet_Z_nonbiased_words()
		
		checkpoint = torch.load('model_all-lang/save_lit_'+ lang+'_rep'+str(rep) +'.pth.tar',map_location ='cpu')['state_dict']
		
		
		for key in list(checkpoint.keys()):
		    if 'module.' in key:
		        checkpoint[key.replace('module.', '')] = checkpoint[key]
		        del checkpoint[key]
		net.load_state_dict(checkpoint)
		# net.eval()
		
		nBli = {}; #nBli['v1'] = []; nBli['v2'] = [];  nBli['v4'] = [];   nBli['it'] = []; 
		nBli['h'] = [];
		for i in range(12):
		    stimtemp, classes = next(dataiter)
		    _, _, _, _, varh, _ = net(stimtemp.float())
		    # nBli['v1'].extend(varv1.detach().numpy())
		    # nBli['v2'].extend(varv2.detach().numpy())
		    # nBli['v4'].extend(varv4.detach().numpy())
		    nBli['h'].extend(varh.detach().numpy())
		    # nBli['it'].extend(varit.detach().numpy())
	# 	    print(i)
		
		#% Identify word selective units based on 3 standard deviations above the mean
		data = np.array(nBli['h'])
		all_neuid = []
		for Lid,ql in enumerate(Lidx_start):
			neuid = []; 
	
			for unit in range(np.size(data,1)):
				Objmean   = np.mean(data[qo,unit])
				Objstdev  = np.var(data[qo,unit])**0.5
				Lmean = np.mean(data[ql:ql+400,unit])
		    
				if np.var(data[ql:ql+400,unit]) > 1:
					if Lmean > Objmean + 3*Objstdev:
						neuid.append(unit)
						
			all_neuid.extend(neuid)			
			nWS[Lid, Mid, rep] = len(neuid)
		
	# 	print(nWS)
		nWS_unique[Mid, rep] = len(np.unique(all_neuid))
	
#%%
ws_avg = np.mean(nWS,2)
ws_std = np.std(nWS,2)

xpos = [0.8,0.9,1,1.1,1.2]
for i in range(5):
	plt.bar(np.arange(8)+xpos[i],ws_avg[i,:],yerr = ws_std[i,:],width = 0.1)
plt.xticks(np.arange(1,9),labels = allmodels);
plt.legend(['FR','EN','CH','Tel','Mal']);
plt.ylabel('Number of word-selective units')
# plt.savefig('word_selective_units.pdf')
# plt.close()

#%% 
# np.savez('word_selective_units_data',nWS = nWS, nWS_unique = nWS_unique)
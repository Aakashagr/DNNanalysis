# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:57:42 2023

@author: Aakash
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle
#%%
matrix_cat = {}

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
	if 'module.' in key:
		checkpoint[key.replace('module.', '')] = checkpoint[key]
		del checkpoint[key]
net.load_state_dict(checkpoint)

# layer = 'h'
for layer in ['v2','v4','it','h']:
# for z in range(1):	
	with open('WSunits/rep0/WSunits_lit_'+layer+'.pkl', 'rb') as f:
	 	wordSelUnit = pickle.load(f)
	
	qcounts = []
	
	for i, nid in enumerate(wordSelUnit):
		folderid = 'WSunit_' + str(i).zfill(3)
		print(folderid)
	
		data_dir = 'stimuli/PosTuning_allunits_'+layer+'/'
		transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
		
		chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
		dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
		dataiter = iter(dataloaders[folderid])
		
		nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; 		
		for i in range(1):
			stimtemp, classes = next(dataiter)
			nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'], _ = net(stimtemp.float())
	# 		varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
	
		Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
		Arel = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
		
		x = np.array(nBli[layer].detach().numpy())
		out = x[:20,nid]
	
		# excluding units with very low activation	
		if np.max(out) < 5:
			qcounts.append([-1,-1,-1,-1,-1])
			continue
		
		matrix = np.zeros((5,4))
		for p in range(20):
			matrix[Arel[p], Rrel[p]] = out[p]
			
		thresh = max(6, (max(out) - min(out))*.3)
		
		matrix[matrix < thresh] = 0
		matrix[matrix >= thresh] = 1
		
		rows, cols = 5, 4
		row_count, col_count, diagonal_count, mix_count, colid = 0, 0, 0, 0, -1
	
	    # Count entries along rows, columns, diagonal, and mix
		for i in range(rows):
			for j in range(cols):
				if matrix[i][j] != 0:
					if i + 1 < rows and matrix[i][j] == matrix[i + 1][j]:
						col_count += 1
					if j + 1 < cols and matrix[i][j] == matrix[i][j + 1]:
						row_count += 1
					if i + 1 < rows and j + 1 < cols and matrix[i][j] == matrix[i + 1][j + 1]:
						diagonal_count += 1
					if i + 1 < rows and j - 1 >= 0 and matrix[i][j] == matrix[i + 1][j - 1]:
						mix_count += 1
					else:
						colid = j
		
		qcounts.append([row_count, col_count, diagonal_count, mix_count, colid])
	
	qcounts = np.array(qcounts)
	matrix_cat[layer] = qcounts
	
	np.save('plots/matrix_type/postuning_'+layer+'.npy',qcounts)
# %%
layer = 'h'
qcounts = np.load('plots/matrix_type/postuning_'+layer+'.npy')
qcounts = np.delete(qcounts,np.where(qcounts[:,1] == -1)[0], axis = 0)
# qcounts = np.delete(qcounts,np.where(qcounts[:,1] == 16)[0], axis = 0)


# maxid = np.argmax(qcounts[:,:4],1)
# len(np.where(maxid == 3)[0])/len(maxid)

# maxid = np.where((qcounts[:,3] >= np.max(qcounts[:,:3], axis = 1)) & (qcounts[:,3] != 0))[0]

# for ordinal  units
# maxid = np.where((qcounts[:,1] >= np.max(qcounts[:,[0,2,3]], axis = 1)) & (qcounts[:,1] != 16))[0]

# For edge coding units
maxid = np.where((np.max(qcounts[:,[0,2,3]], axis = 1) == 0))[0]

len(maxid)/len(qcounts[:,0])

# nq = np.where(qcounts)
# pp = np.where((qcounts[:,1] == 4) & (qcounts[:,0] == 0))
	

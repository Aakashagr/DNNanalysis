# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:53:57 2023

@author: Aakash
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import torch
import os
import torchvision.transforms as transforms
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets
import pickle
#%%

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
	if 'module.' in key:
		checkpoint[key.replace('module.', '')] = checkpoint[key]
		del checkpoint[key]
net.load_state_dict(checkpoint)
net.eval()

#%% Identifying word selective units

with open('WSunits/rep0/WSunits_lit_v2.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)
	 
with open('WSunits/rep0/WSunits_lit_v1.pkl', 'rb') as f:
 	wordSelUnitv1 = pickle.load(f)
	 
#%% Top response inputs to the V2 layer

lidx = 84290
# lidx = 7459
qunit = np.where(np.array(wordSelUnit) == lidx)[0][0]+1

folderid = 'WSunit_' + str(qunit).zfill(3)
print(folderid)
data_dir = 'stimuli/PosTuning_allunits_v2/'
transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
dataiter = iter(dataloaders[folderid])

stimtemp, classes = next(dataiter)
varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
v1resp = np.max(varv1[:20,:].detach().numpy(),axis = 0)

#% Identifying all word selective units in the given filter

nfilt = 128; fsize = 28;  
[z_out,x_out,y_out] = np.unravel_index(lidx, (nfilt, fsize, fsize))
weights = checkpoint['V2.conv.weight']
weights_v1 = checkpoint['V1.conv.weight']


# using sum instead of norm as all responses are more than zero after Relu
wnorm = [sum(weights[z_out,i,:,:].flatten()).item() for i in range(weights.shape[1])] 

x_in = x_out * 2 
y_in = y_out * 2  

nfilt = int(nfilt/2); fsize = fsize*2
index = []; RFindices = {}
index_range = np.zeros((nfilt,25))
offset = [-2,-1,0,1,2]

peak_resp = np.zeros((nfilt))
for i in range(nfilt):
	index.append(np.ravel_multi_index((i, x_in, y_in), dims=(nfilt, fsize, fsize)))

	for k in range(len(offset)):
		index_range[i,k] = index[i]+offset[k] - 2*fsize
		index_range[i,k+5] = index[i]+offset[k] - fsize
		index_range[i,k+10] = index[i]+offset[k] 
		index_range[i,k+15] = index[i]+offset[k] + fsize
		index_range[i,k+20] = index[i]+offset[k] + 2*fsize

	qindex = [int(x) for x in index_range[i,:]]
	peak_resp[i] = max(v1resp[qindex])
	
top_inp = np.argsort(-np.array(wnorm)*peak_resp)[:5]

# fig, axes = plt.subplots(8, 8, figsize=(10,10))
# for i, ax in enumerate(axes.flat):
#  	qtemp = np.transpose(weights_v1[i,:,:,:],[2,1,0])
#  	qtemp = qtemp - min(qtemp.flatten())
#  	qtemp = qtemp/max(qtemp.flatten())
 	  
#  	ax.imshow(qtemp);
#  	ax.axis("off")  # Turn off axis labels
# fig.savefig('plots/connectivity/v1filter.pdf')

	
#%
nsize = {}; nsize['v1'] = 56*56; nsize['v2'] = 784; nsize['v4'] = 196; nsize['it'] = 49; 
sunit = lidx//nsize['v2']

num_images = 5; rows = 2; columns = num_images
fig, axes = plt.subplots(rows, columns, figsize=(5, 2))

# Iterate through the images and plot them in the subplots
maxval = weights[sunit,top_inp,:,:].max().tolist()
for i, ax in enumerate(axes.flat):
	if i < num_images:
		ax.imshow(weights[sunit,top_inp[i],:,:], cmap = 'bwr',vmin=-maxval, vmax=maxval);
		ax.axis("off")  # Turn off axis labels
	else:
		imid = i-num_images
		image = np.transpose(weights_v1[top_inp[imid],:,:,:],[2,1,0])
		image = image - min(image.flatten())
		image = image/max(image.flatten())
		ax.imshow(image); ax.axis('off')

plt.tight_layout()
plt.show()
# fig.savefig('plots/connectivity/v2_'+str(lidx)+ '_pos.pdf')
plt.close(fig)

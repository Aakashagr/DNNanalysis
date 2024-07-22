# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:28:51 2024

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

#%% Identifying word selective units

with open('WSunits/rep0/WSunits_lit_v4.pkl', 'rb') as f:
 	wordSelUnitv4 = pickle.load(f)
	 
with open('WSunits/rep0/WSunits_lit_v2.pkl', 'rb') as f:
 	wordSelUnitv2 = pickle.load(f)
	 
#%% Top response inputs to the V4 layer

# lidx = 1671;
lidx = 4219;

qunit = np.where(np.array(wordSelUnitv4) == lidx)[0][0]

folderid = 'WSunit_' + str(qunit).zfill(3)
print(folderid)
data_dir = 'stimuli/PosTuning_allunits_v4/'
transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
dataiter = iter(dataloaders[folderid])
stimtemp, classes = next(dataiter)

net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
	if 'module.' in key:
		checkpoint[key.replace('module.', '')] = checkpoint[key]
		del checkpoint[key]
net.load_state_dict(checkpoint)
net.eval()
varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())

# Selecting first 20 stimuli to ignore letter spacing and other pairs
WSv2resp = np.max(varv2[:20,wordSelUnitv2].detach().numpy(),axis = 0) 


#%% Identifying all word selective units in the given receptive field of the chosen unit

nfilt = 256; fsize = 14;  
[z_out,x_out,y_out] = np.unravel_index(lidx, (nfilt, fsize, fsize))
weights = checkpoint['V4.conv.weight']

# using sum instead of norm as all responses are more than zero after Relu
wnorm = [sum(weights[z_out,i,:,:].flatten()).item() for i in range(weights.shape[1])] 
wnorm = np.array(wnorm)

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

	RFindices[i] = [index for index, element in enumerate(wordSelUnitv2) if element in np.unique(index_range[i,:])]
	if len(RFindices[i]) > 3:
		peak_resp[i] = max(WSv2resp[RFindices[i]])
	
top_inp = np.argsort(-wnorm*peak_resp)[:5]
bot_inp = np.flip(np.argsort(-wnorm*peak_resp)[-5:])

for i in range(5):
    print(RFindices[top_inp[i]])
print('Inhibit response')
for i in range(5):
    print(RFindices[bot_inp[i]])
	
print((wnorm*peak_resp)[top_inp]); 
print((wnorm*peak_resp)[bot_inp])

#%% Plotting the data
nsize = {}; nsize['v1'] = 56*56; nsize['v2'] = 784; nsize['v4'] = 196; nsize['it'] = 49; 
# weights = checkpoint['V4.conv.weight']
sunit = lidx//nsize['v4']

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
		image = plt.imread('plots/Activation_maximization_lucent/rep0/literate/AM_V2/'+
			 str(top_inp[imid]).zfill(3) +'.png')
		
		image = image[82:142,224+82:-82,:]
		ax.imshow(image); ax.axis('off')

plt.tight_layout()
plt.show()
fig.savefig('plots/connectivity/v4_'+str(lidx)+ '_top.svg')
plt.close(fig)

# Iterate through the images and plot them in the subplots
fig, axes = plt.subplots(rows, columns, figsize=(5, 2))
maxval = weights[sunit,list(bot_inp),:,:].max().tolist()
for i, ax in enumerate(axes.flat):
	if i < num_images:
		ax.imshow(weights[sunit,bot_inp[i],:,:], cmap = 'bwr',vmin=-maxval, vmax=maxval);
		ax.axis("off")  # Turn off axis labels
	else:
		imid = i-num_images
		image = plt.imread('plots/Activation_maximization_lucent/rep0/literate/AM_V2/'+
			 str(bot_inp[imid]).zfill(3) +'.png')
		
		image = image[82:142,224+82:-82,:]
		ax.imshow(image); ax.axis('off')

plt.tight_layout()
plt.show()
fig.savefig('plots/connectivity/v4_'+str(lidx)+ '_bottom.svg')
plt.close(fig)


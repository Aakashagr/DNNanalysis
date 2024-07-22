# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:42:00 2023

Recreating all the plots for any particular unit 

@author: Aakash
"""

layer = 'v4'; unit = 18569


import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import string
from clean_cornets import CORNet_Z_nonbiased_words
from torchvision import datasets

#%% Load network
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)


#%% Load stimulus
data_dir = 'stimuli/PosTuning_letters/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in ['train']}
dataiter = iter(dataloaders['train'])
    
#%%
nBli = {}; nBli['v1'] = [];  nBli['v2'] = [];  nBli['v4'] = []; nBli['it'] = []; nBli['h'] = []; 
for i in range(2):
    stimtemp, classes = next(dataiter)
    # nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    varv1,varv2,varv4,varit,varh, varOut = net(stimtemp.float())
    nBli['v1'].extend(varv1.detach().numpy())
    nBli['v2'].extend(varv2.detach().numpy())
    nBli['v4'].extend(varv4.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['it'].extend(varit.detach().numpy())
    
#%%

out = np.array(nBli[layer])[:,unit]
charcoef = np.reshape(out,[26,8])
meancoeff = np.max(charcoef,1)
        
#%% Generating position tuning stimuli for the given unit
import copy
from PIL import Image, ImageDraw, ImageFont
import os, gc

def gen2(savepath='', text = 'text', index = 0,fontname='Arial', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0):
    if upper:
        text = text.upper()
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')
    img.save(savepath+text+str(index)+'.jpg')


letters = string.ascii_uppercase
path_out='stimuli/PosTuning_allunits_others/'

    
q1 = np.argsort(meancoeff)
T = [q1[-1], q1[-2]]
D = [q1[0], q1[1]]
temp = list('        ')
wordlist = []
for t,d in zip(T,D):
	for sp in range(5): # Spacing 
		for tp in range(4): # Target position
			stim = copy.deepcopy(temp)
			for dp in range(4): # 4 letters
				if dp == tp:
					stim[dp+sp] = letters[t]
				else:
					stim[dp+sp] = letters[d]
			wordlist.append("".join(stim))
        
	# Letter spacing stimuli
	for sp in range(4): # Spacing 
		for tp in np.arange(0,6,2): # Target position
			stim = copy.deepcopy(temp)
			for dp in np.arange(0,6,2): # 4 letters`
				if dp == tp:
					stim[dp+sp] = letters[t]
				else:
					stim[dp+sp] = letters[d]
			wordlist.append("".join(stim))
                
#create train and val folders
unum = 'Layer_'+layer+'_WSunit_' + str(unit).zfill(3)
for n,f in enumerate(wordlist):
	target_path = path_out+unum+'/'+str(n).zfill(2)+'_'+f+'_'
	os.makedirs(target_path)

#for each word, create num_train + num_val exemplars, then split randomly into train and val.
for i,w in enumerate(wordlist):
	gc.collect()
	path = path_out+unum+'/'+str(i).zfill(2)+'_'+w+'_/'
	gen2(savepath=path, text=w, index=0, fontname='Consolas', size=60, xshift=0, yshift=0, upper=1)

#%%
folderid = unum
data_dir = 'stimuli/PosTuning_allunits_others/'
transform = {folderid: transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}
    
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in [folderid]}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 200,shuffle = False) for x in [folderid]}
dataiter = iter(dataloaders[folderid])
stimtemp, classes = next(dataiter)
nBli['v1'], nBli['v2'], nBli['v4'], nBli['it'], nBli['h'],  nBli['out'] = net(stimtemp.float())
    
#%% Generating response profile for the chosen unit
wordlist = [i[3:] for i in chosen_datasets[folderid].classes]
Rrel = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,0,1,2,0,1,2,0,1,2]
Arel = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,1,1,1,2,2,2,3,3,3]
Rrel = np.tile(Rrel,2).flatten(); 
Arel = np.tile(Arel,2).flatten(); 

x = np.array(nBli[layer].detach().numpy())
out = x[:,unit]
    
fig, axs = plt.subplots(2,2, figsize=(7,9), facecolor='w', edgecolor='k')
axs = axs.ravel();
sid = {}; 
sid[0] = np.arange(0,20); sid[1] = np.arange(20,32)
sid[2] = np.arange(32,52); sid[3] = np.arange(52,64)

for i in range(4):
	qmat = np.zeros((5,4))
	for p in sid[i]:
		qmat[Arel[p], Rrel[p]] = out[p]
	im = axs[i].pcolor(qmat, cmap = 'Blues')
	axs[i].set_xticks(ticks = np.arange(4)+0.5); axs[i].set_xticklabels(['1','2','3','4']);
	axs[i].set_yticks(ticks = np.arange(5)+0.5); axs[i].set_yticklabels(['1','2','3','4','5']);
	axs[i].set_xlabel('Ordinal letter position')
	axs[i].set_ylabel('Absolute word position')
	axs[i].set_title('Unit ID#: ' + str(unit) + ', stim: ' + wordlist[sid[i][0]])
	im.set_clim(0,max(out)) 
       
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)    
    
fig.savefig('plots/PosTuning_mix/' +folderid + '.pdf')
fig.savefig('plots/PosTuning_mix/' +folderid + '.png')
plt.close(fig)

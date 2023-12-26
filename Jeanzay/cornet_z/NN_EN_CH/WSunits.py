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
from torchvision import datasets, transforms
from training_code import clean_cornets

#%%

data_dir = 'stimuli/wordselective_stimuli/'
transform = {'train': transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform =  transform[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size = 80,shuffle = False) for x in ['train']}

#%%
dataiter = iter(dataloaders['train'])

nBli = {}; nBli['v1'] = []; nBli['v2'] = []; nBli['v4'] = [];
nBli['it'] = []; nBli['h'] = []; nBli['out'] = []

literate = 1
if literate:
	net = clean_cornets.CORNet_Z_nonbiased_words()
	checkpoint = torch.load('models/save_lit_no_bias_z_79_full_nomir.pth.tar', map_location=torch.device('cpu'))['state_dict']
else:
	net = clean_cornets.CORnet_Z_tweak()
	checkpoint = torch.load('models/save_illit_z_79_full_nomir.pth.tar', map_location=torch.device('cpu'))['state_dict']

for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)
net.eval()

for i in range(15):
    stimtemp, classes = next(dataiter)
    varV1, varV2, varV4, varIt, varh, varOut = net(stimtemp.float())
    nBli['v1'].extend(varV1.detach().numpy())
    nBli['v2'].extend(varV2.detach().numpy())
    nBli['v4'].extend(varV4.detach().numpy())
    nBli['it'].extend(varIt.detach().numpy())
    nBli['h'].extend(varh.detach().numpy())
    nBli['out'].extend(varOut.detach().numpy())
    print(i)

#%% Identify word selective units based on 3 standard deviations above the mean of object responses
qo = np.array(np.arange(0,400));
qe = np.array(np.arange(400,800));
qc = np.array(np.arange(800,1200));
data_d = np.array(nBli['out'])

neuid_c = []; neuid_e = [];
for unit in range(np.size(data_d,1)):
    Enmean = np.mean(data_d[qe,unit])
    Chmean = np.mean(data_d[qc,unit])
    Objmean   = np.mean(data_d[qo,unit])
    Objstdev  = np.var(data_d[qo,unit])**0.5

    if Enmean >= Objmean + 3*Objstdev:
        neuid_e.append(unit)

    if Chmean >= Objmean + 3*Objstdev:
        neuid_c.append(unit)

#%% Units selective to either of the languages
neuid_ec = np.unique(np.concatenate([neuid_c,neuid_e]))
neuid_c2 = []; neuid_e2 = [];

for unit in neuid_ec:
	Enmean = np.mean(data_d[qe,unit])
	Chmean = np.mean(data_d[qc,unit])
	Enstdev = np.var(data_d[qe,unit])**0.5
	Chstdev = np.var(data_d[qc,unit])**0.5

	if Enmean >= Chmean + 3*Chstdev:
		neuid_e2.append(unit)

	if Chmean >= Enmean + 3*Enstdev:
		neuid_c2.append(unit)

print(['% word-selective units: ' + str(np.size(neuid_ec)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_ec))])
print(['% of English word-selective units: ' + str(np.size(neuid_e2)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_e2))])
print(['% of Chinese word-selective units: '+ str(np.size(neuid_c2)/np.size(data_d,1)) + ', count: ' + str(np.size(neuid_c2))])


#%%
# from matplotlib_venn import venn2

# venn2(subsets = (len(neuid_c2), len(neuid_e2), len(neuid_ec) - len(neuid_c2) - len(neuid_e2)), set_labels = ('Chinese', 'English'))
# plt.title('Number of word selective units')
# plt.show()
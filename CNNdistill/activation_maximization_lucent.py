# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:06:02 2022

@author: Aakash
"""

import torch
from lucent.optvis import render, param, transform, objectives
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
#%%

# net = CORnet_Z_tweak()
# checkpoint = torch.load('models/rep1/save_illit_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
net = CORNet_Z_nonbiased_words()
checkpoint = torch.load('models/rep1/save_lit_no_bias_z_79_full_nomir.pth.tar',map_location ='cpu')['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
net.load_state_dict(checkpoint)

net.to(device).eval()

# _ = render.render_vis(net, "IT_output:31")

#%%
param_f = lambda: param.image(224, batch=2)
# obj = objectives.channel("IT_output", 63, batch=1) - objectives.channel("IT_output", 63, batch=0)
fname = 'plots/AM_V1/'
os.makedirs(fname,exist_ok=True)
for C in range(64):
    X = None; Y = None;
    obj = objectives.neuron("V1_conv", C,x=X,y=Y,batch=1) - objectives.neuron("V1_conv", C,x=X,y=Y, batch=0)
    _ = render.render_vis(net, obj, param_f, thresholds = [1000],save_image= True ,image_name=fname+ str(C).zfill(3)+'.png')
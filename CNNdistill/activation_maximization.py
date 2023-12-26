#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:45:38 2022

@author: aakash
"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PytorchRevelio import PytorchRevelio
from utilities_PytorchRevelio import imagenet_labels
from clean_cornets import CORNet_Z_nonbiased_words, CORnet_Z_tweak

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

#%%

# choose GPU if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))

# put network on device
net.to(device)

# print name of modules
# for key, value in PytorchRevelio.layers_name_type(net):
#     print('+' * 10)
#     print(key)
#     print('-' * 10)
#     print(value)

# network transformer for input image
img_size = (224, 224, 3)
img_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# for different convolutional filter and neuron in fully connected layer
# show representation
first_layer_name = 'V1.conv'
layer_name = 'V2.conv'


# select convolutional and fully connected layers for visualization
layer = PytorchRevelio.return_module_by_name(network=net, module_name=layer_name)
filter_neuron_num = layer.out_channels
layer_type = 'Conv2d'
num_iter = 500
lr = 0.009
start_sigma_color = 25
end_sigma_color = 110
start_sigma_space = 25
end_sigma_space = 110
kernel_size = 3


num_iter = 500
lr = 10
start_sigma_color = 25
end_sigma_color = 110
start_sigma_space = 25
end_sigma_space = 110
  
            
# for each selected filter or neuron, calculate representation
IMG = []
for i in range(128):
    img = PytorchRevelio.activation_maximization_with_bilateral_blurring(
        network= net,
        img_transformer=img_transformer,
        in_img_size=img_size,
        first_layer_name=first_layer_name,
        layer_name=layer_name,
        filter_or_neuron_index=i,
        num_iter=num_iter,
        start_sigma_color=start_sigma_color,
        end_sigma_color=end_sigma_color,
        start_sigma_space=start_sigma_space,
        end_sigma_space=end_sigma_space,
        kernel_size=kernel_size,
        lr=lr,
        device=device)
    
    # to cpu and normalize for illustration purpose
    img = PytorchRevelio.tensor_outputs_to_image(img)
    IMG.append(img)
    print(i)

#%%
plt.figure(figsize=(35,20))
for i in np.arange(0,128):
    ax = plt.subplot(8, 16, i+1)
    plt.imshow(IMG[i])
    plt.suptitle('Layer Name: {}'.format(layer_name))
    ax.axis('off')
    print('Processing of layer {}, filter/neuron {} is done.'.format(layer_name, i))

plt.show()

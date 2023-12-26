# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:13:54 2023

@author: Aakash
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, gc
from tqdm import tqdm
from unidecode import unidecode


words = []
f = open("multiwords.txt", "r", encoding = 'utf8')
lines = f.readlines()

for l in lines:
	words.append(unidecode(l).strip())
 
#%%
path_out='words_malabi/'
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

wordlist = words

#create train and val folders
for m in ['train']:
	for f in wordlist[::2]:
		target_path = path_out+m+'/'+f
		os.makedirs(target_path, exist_ok=True)

#for each word, create num_train + num_val exemplars, then split randomly into train and val.
for i in np.arange(0,len(wordlist)-1,2):
	gc.collect()
	text = wordlist[i].upper()
	print(i)
	path = path_out+'train/'+text+'/'
	
# 		gen2(savepath=path, text=w, index=0, fontname='arial', size=60, xshift=0, yshift=0, upper=1)
	
	W = 500
	H = 500
	text = wordlist[i].upper()

	for cix,cval in enumerate([0,64,128,255]):
		img = Image.new("RGB", (W,H), color = (255, 255, 255))
		fnt = ImageFont.truetype('arial.ttf', 60)
		draw = ImageDraw.Draw(img)
		w, h = fnt.getsize(text)
		draw.text((W/2-w-10, (H-h)/2), text, font=fnt, fill = (0,0,0))
		draw.text((W/2+10, (H-h)/2), wordlist[i+1].upper(), font=fnt, fill = (cval, cval, cval))
		img.save(path+text+str(cix)+'.jpg')



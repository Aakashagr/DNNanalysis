# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:01 2022

@author: Aakash
"""

# Stim generation
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, gc
import copy
import pandas as pd
import random
random.seed(42)

wstim = [   'WAWO', 'WAPE', 'AWEP', 'AWAW',
 			'WEWE', 'WEWO', 'APOP', 'APAL',
 			'WOLE', 'WOPA', 'ALAW', 'ALAP',
 			'PALO', 'PALE', 'EWOL', 'EWEL',
 			'PEPO', 'PELA', 'EPOW', 'EPOL',
 			'POPE', 'POWA', 'ELAP', 'ELEP',
 			'LAPA', 'LALO', 'OWEL', 'OWEW',
 			'LEWA', 'LEPO', 'OPAL', 'OPOP',
 			'LOLA', 'LOWE', 'OLEW', 'OLOW',
 			]

# Definition ot create the stimuli
def gen2(savepath='', text = 'text', index=1, fontname='Arial', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0):
	if upper:
		text = text.upper()
	img = Image.new("RGB", (W,H), color = (255, 255, 255))
	fnt = ImageFont.truetype(fontname+'.ttf', size)
	draw = ImageDraw.Draw(img)
	w, h = fnt.getsize(text)
	draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')
	img.save(savepath+str(index).zfill(3) +'_'+text.replace(" ","")+'.jpg')

# Word stimuli
def CreateExpWords(path_out='stimuli/neural_stim/train/'):
	os.makedirs(path_out, exist_ok=True)
	
	temp = list('        ')
	stimlist = []
	for s in wstim:
		for i in range(5):
			stim = copy.deepcopy(temp)
			stim[i:i+4] = s
			stimlist.append("".join(stim))
			
	for i,w in enumerate(stimlist):
		gc.collect()
		print (w,)
		gen2(savepath=path_out, text=w, index=i, fontname='Consolas', size=60, xshift=0, yshift=0, upper=1)

	return 'done'

CreateExpWords()



# Non word stimuli
df = pd.read_excel('MTS_stimuli.xlsx')
allstim = df.to_numpy()
nwstim = []
for i in range(36):
	for j in range(8):
		nwstim.append(allstim[i][j+1])
		
def CreateExpNonWords(path_out='stimuli/neural_stim/train/'):
 	os.makedirs(path_out, exist_ok=True)
 							
 	for i,w in enumerate(nwstim):
		 gc.collect()
		 print (w,)
		 gen2(savepath=path_out, text=w, index=i+180, fontname='Consolas', size=60, xshift=0, yshift=0, upper=1)

 	return 'done'

CreateExpNonWords()

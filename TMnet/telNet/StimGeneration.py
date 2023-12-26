# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:01 2022

@author: Aakash
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os, gc
from tqdm import tqdm
import random
#%%

def gen2(savepath='', text = 'text', index=1, mirror=False, invert=False, fontname='Arial', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0, show=None):
	if upper:
		text = text.upper()
	if invert:
		text = text[::-1]
	img = Image.new("RGB", (W,H), color = (255, 255, 255))
	#fnt = ImageFont.truetype('/Library/Fonts/'+fontname+'.ttf', size) #size in pixels
	fnt = ImageFont.truetype(fontname+'.ttf', size)
	draw = ImageDraw.Draw(img)
	w, h = fnt.getsize(text)
	draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')

	if mirror:
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
	if savepath != '':
		img.save(savepath+text+str(index)+'.jpg')
	if show:
		img.save('plots/'+text+str(index)+'.jpg')
		img.show()
	if savepath == '':
		print('I was here')
		img.show()
		return img

#%%  generating stim for word selective units
def CreateMiniWordSet(path_out='stimuli/telwords/',num_train=10, num_val=0):
	#define words, sizes, fonts
	words = []
	with open('Telugu_words.txt', encoding="utf8") as f:
		for line in f:
			words.append(line.strip())
	wordlist = words[1:1000:25]
		
	sizes = [30, 35, 40, 45, 50, 55, 60, 65, 70]
	fonts = ['Nirmala','NotoSansTelugu']
	xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
	yshift = [-30, -15, 0, 15, 30]
	
	#create train and val folders 
	for m in ['train', 'val']:
		for f in wordlist:
			target_path = path_out+m+'/'+f
			os.makedirs(target_path)
	
	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for w in tqdm(wordlist):
		gc.collect()
		print (w,)
		for n in range(num_train + num_val):
			if n < num_train:
				path = path_out+'train/'+w+'/'
			else:
				path = path_out+'val/'+w+'/'
			
			f = random.choice(fonts)
			s = random.choice(sizes)
			u = random.choice([0,1])
			x = random.choice(xshift)
			y = random.choice(yshift)
			gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)

	return 'done'


CreateMiniWordSet()
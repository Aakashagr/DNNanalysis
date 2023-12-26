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
def CreateMiniWordSet(path_out='stimuli/bigrams/',num_train=1, num_val=0):
	#define words, sizes, fonts
	words = []
	with open('bigram_list.txt', encoding="utf8") as f:
		for line in f:
			words.append(line.strip())
	wordlist = words
		
	sizes = [50]
	fonts = 'Nirmala'
	xshift = [0]
	yshift = [0]
	
	#create train and val folders 
	for m in ['train']:
		for i, f in enumerate(wordlist):
			target_path = path_out+m+'/'+ str(i).zfill(2)+'_'+f
			os.makedirs(target_path, exist_ok= True)
	
	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for i,w in enumerate(wordlist):
		gc.collect()
		print (w,)
		path = path_out+'train/'+ str(i).zfill(2)+'_'+w+'/'		
		f = fonts
		s = random.choice(sizes)
		x = random.choice(xshift)
		y = random.choice(yshift)
		gen2(savepath=path, text=w, index=1, fontname=f, size=s, xshift=x, yshift=y, upper=1)

	return 'done'


CreateMiniWordSet()

#%%  generating stim for word selective units
def CreateMiniWordSet(path_out='stimuli/bigrams_fmri/',num_train=1, num_val=0):
	#define words, sizes, fonts
	words = []
	with open('bigrams_fmri.txt', encoding="utf8") as f:
		for line in f:
			words.append(line.strip())
	wordlist = words
		
	sizes = [50]
	fonts = 'Nirmala'
	xshift = [0]
	yshift = [0]
	
	#create train and val folders 
	for m in ['train']:
		for i, f in enumerate(wordlist):
			target_path = path_out+m+'/'+ str(i).zfill(2)+'_'+f
			os.makedirs(target_path, exist_ok= True)
	
	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for i,w in enumerate(wordlist):
		gc.collect()
		print (w,)
		path = path_out+'train/'+ str(i).zfill(2)+'_'+w+'/'		
		f = fonts
		s = random.choice(sizes)
		x = random.choice(xshift)
		y = random.choice(yshift)
		gen2(savepath=path, text=w, index=1, fontname=f, size=s, xshift=x, yshift=y, upper=1)

	return 'done'


CreateMiniWordSet()
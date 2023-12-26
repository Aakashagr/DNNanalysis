# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:01 2022

@author: Aakash
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, gc
from tqdm import tqdm
from scipy.io import loadmat

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


#%%  FMRI stim
def CreateExpWords(path_out='stimuli/meg_stim/'):

	mat = loadmat('stiminfo.mat')['stiminfo']
	qwordlist = []
	for i in range(80):
		qwordlist.append(mat[i,0][0])

	for i in range(80):
		qwordlist.append(mat[i,1][0])

	wordlist = []
	for i in range(160):
		wordlist.append(qwordlist[i]+'      ')
	for i in range(160):
		wordlist.append('   '+ qwordlist[i]+'   ')
	for i in range(160):
		wordlist.append('      '+qwordlist[i])

	sizes = [60]
	fonts = ['Consolas']
	xshift = [0]
	yshift = [0]

	#create train and val folders
	for m in ['train']:
		for n,f in enumerate(wordlist):
			target_path = path_out+m+'/'+str(n).zfill(3)
			os.makedirs(target_path, exist_ok=True)

	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for i,w in enumerate(wordlist):
		gc.collect()
		print (w,)
		path = path_out+'train/'+str(i).zfill(3)+'/'
		n = 0
		for f in fonts:
			for s in sizes:
				for x in xshift:
					for y in yshift:
						n = n+1
						gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

	return 'done'

CreateExpWords()
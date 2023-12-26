# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:01 2022

@author: Aakash
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, gc
from tqdm import tqdm
from unidecode import unidecode


words_malabi = []
f = open("malabi_stimuli.txt", "r", encoding = 'utf8')
lines = f.readlines()

for l in lines:
	words_malabi.append(unidecode(l).strip())


words = np.load('wordlist10k.npy')
words = np.unique(words)

words2 = []
for w in words:
	if w not in words_malabi:
		words2.append(w)		

words = np.array(words2)[np.ceil(np.arange(0, len(words2)-1, len(words2)/9693)).astype(int)]   
wordsall = np.r_[words,np.unique(words_malabi)]
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

#%%
def CreateMiniWordSet(path_out='words/',num_train=1):
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

	wordlist = wordsall


# 	sizes = 60
	fonts = ['arial']
# 	xshift = 0
# 	yshift = 0

	#create train and val folders
	for m in ['train']:
		for f in wordlist:
			target_path = path_out+m+'/'+f
			os.makedirs(target_path)

	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for w in tqdm(wordlist):
		gc.collect()
		print (w,)
		path = path_out+'train/'+w+'/'
		n = 0
		for f in fonts:
			# for s in sizes:
			#	 for x in xshift:
			#		 for y in yshift:
				n = n+1
				gen2(savepath=path, text=w, index=n, fontname=f, size=60, xshift=0, yshift=0, upper=1)

	return 'done'

CreateMiniWordSet()


#%% Test set (different size and font style)
def CreateMiniWordSet(path_out='words/',num_train=1):
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

	wordlist = wordsall

# 	sizes = 50
	fonts = ['calibri']
# 	xshift = 0
# 	yshift = 0

	#create train and val folders
	for m in ['test']:
		for f in wordlist:
			target_path = path_out+m+'/'+f
			os.makedirs(target_path)

	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for w in tqdm(wordlist):
		gc.collect()
		print (w,)
		path = path_out+'test/'+w+'/'
		n = 0
		for f in fonts:
			# for s in sizes:
			#	 for x in xshift:
			#		 for y in yshift:
				n = n+1
				gen2(savepath=path, text=w, index=n, fontname=f, size=60, xshift=0, yshift=0, upper=1)

	return 'done'

CreateMiniWordSet()


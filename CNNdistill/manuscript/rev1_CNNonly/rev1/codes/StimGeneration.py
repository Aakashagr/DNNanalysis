# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:16:01 2022

@author: Aakash
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, gc
from tqdm import tqdm

#%%

def gen2(savepath='', text = 'text', index=1, mirror=False, invert=False, fontname='Arial', W = 500, H = 500, size=24, xshift=0, yshift=0, upper=0, show=None):
	if upper:
		text = text.upper()
	if invert:
		text = text[::-1]
	img = Image.new("RGB", (W,H), color = (255, 255, 255))
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

#%% Create all French words stimuli at 8 different locations and sizes
def CreateMiniWordSet(path_out='stimuli/wordsets_1000cat_8ex/',num_train=100):
	wordlist = np.load('FR_wordlist.npy')


	sizes = [40,80]
	fonts = ['arial']
	xshift = [-30, 30]
	yshift = [-15,15]

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
			for s in sizes:
				for x in xshift:
					for y in yshift:
						n = n+1
						gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

	return 'done'

CreateMiniWordSet()

#%% Create french word list at the centre
def CreateMiniWordSet(path_out='stimuli/wordsets_1000cat/',num_train=100):

	wordlist = np.load('FR_wordlist.npy')


	sizes = [60]
	fonts = ['arial']
	xshift = [0]
	yshift = [0]

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
			for s in sizes:
				for x in xshift:
					for y in yshift:
						n = n+1
						gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

	return 'done'

CreateMiniWordSet()

#%% Classification analysis stimuli 
import random
common_words = np.intersect1d(np.load('fr_wordlist.npy'), np.load('en_wordlist.npy'))

def CreateMiniWordSet(path_out='stimuli/EN_FR_classification/'):
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

	for lang in ['FR','EN']:
		wordlist = np.load(lang+'_wordlist.npy')
		
		# excluding words common in english and French (n = 41)
		wordlist = list(set(wordlist) - set(common_words))
	
		sizes = [40, 50, 60, 70, 80]
		xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
		yshift = [-30, -15, 0, 15, 30]	
		fonts = ['arial','times']

	
		#create train and val folders
		for m in ['train']:
			for f in wordlist:
				target_path = path_out+m+'/'+lang+'_'+f
				os.makedirs(target_path, exist_ok=True)
	
		for w in tqdm(wordlist):
			gc.collect()
			print (w,)
			path = path_out+'train/'+lang+'_'+w+'/'
			n = 0
			f = random.choice(fonts)
			s = random.choice(sizes)
			u = random.choice([0,1])
			x = random.choice(xshift)
			y = random.choice(yshift)
			n = n+1
			gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)
	
	return 'done'

CreateMiniWordSet()

#%% Generating position tuning stimuli for all word selective units
# NOTE: we are generating reponses for two stim pairs whereas only one stim pair
# is reported in the paper.
import pickle
import copy
import string
roi = 'v4'
with open('mean_response_coeff/mean_lresponse_'+roi+'.pkl', 'rb') as f:
 	meancoeff = pickle.load(f)

letters = string.ascii_uppercase
path_out='stimuli/PosTuning_allunits_'+roi+'/'

for unit in range(np.shape(meancoeff)[0]):
	print (unit)
	q1 = np.argsort(meancoeff[unit,:])
	T = [q1[-1], q1[-2]]
	D = [q1[0], q1[1]]
	temp = list('		')
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
	unum = 'WSunit_' + str(unit).zfill(3)
	for n,f in enumerate(wordlist):
		target_path = path_out+unum+'/'+str(n).zfill(2)+'_'+f+'_'
		os.makedirs(target_path)

	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	for i,w in enumerate(wordlist):
		gc.collect()
		path = path_out+unum+'/'+str(i).zfill(2)+'_'+w+'_/'
		gen2(savepath=path, text=w, index=0, fontname='Consolas', size=60, xshift=0, yshift=0, upper=1)

#%% Single letter stimuli for determining position coding
import pickle
import copy
import string

letters = string.ascii_uppercase
path_out='stimuli/PosTuning_letters/train/'

temp = list('		')
wordlist = []
for lid in letters:
	for p in range(8):
		stim = copy.deepcopy(temp)
		stim[p] = lid
		wordlist.append("".join(stim))

os.makedirs(path_out, exist_ok = True)

# for each word, create num_train + num_val exemplars, then split randomly into train and val.
for i,w in enumerate(wordlist):
	gc.collect()
	path = path_out+'/'
	gen2(savepath=path + str(i).zfill(3), text=w, index=0, fontname='Consolas', size=60, xshift=0, yshift=0, upper=0)



#%% Bigram stimuli for dissimilarity analysis

def CreateExpWords(path_out='stimuli/bigrams/'):
	
	letters = ['E','A','S','T','I','R','N']

	words = []
	for i in range(7):
		for j in range(7):
			words.append(letters[i] + letters[j])
	
	wordlist = words
	fonts = ['Consolas']

	n = 0
	for m in ['train']:
		for f in wordlist:
			target_path = path_out+m+'/'+str(n).zfill(4)+'_'+f
			os.makedirs(target_path)
			n = n+1
	
	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	n = 0
	for w in tqdm(wordlist):
		path = path_out+'train/'+str(n).zfill(4)+'_'+w+'/'
		gen2(savepath=path, text=w, index=0, fontname=fonts[0], size=40, upper=1)
		n +=1

	return 'done'

CreateExpWords()


#%% stimuli for actvation maximization

def CreateExpWords(path_out='stimuli/AM/'):
	
	wordlist = ['air','pain','square']
	fonts = ['Consolas']

	n = 0
	for m in ['train']:
		for f in wordlist:
			target_path = path_out+m+'/'+str(n).zfill(4)+'_'+f
			os.makedirs(target_path)
			n = n+1
	
	#for each word, create num_train + num_val exemplars, then split randomly into train and val.
	n = 0
	for w in tqdm(wordlist):
		path = path_out+'train/'+str(n).zfill(4)+'_'+w+'/'
		gen2(savepath=path, text=w, index=0, fontname=fonts[0], size=50, upper=1)
		n +=1

	return 'done'

CreateExpWords()

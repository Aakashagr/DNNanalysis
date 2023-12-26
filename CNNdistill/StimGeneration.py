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
def CreateMiniWordSet(path_out='stimuli/wordsets_1000cat_8ex/',num_train=100):
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

    wordlist = np.load('sorted_wordlist.npy')


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

#%%
def CreateMiniWordSet(path_out='stimuli/wordsets_1000cat/',num_train=100):
# def CreateMiniWordSet(path_out='wordsets_1000cat_LR/'):

    wordlist = np.load('sorted_wordlist.npy')


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
#%% Example stimuli to study neural tuning based on model
def CreateExpWords(path_out='stimuli/neuraltuning/',num_train=100):

    wordlist = ['SSSSSSSS','XXSSSSXX','SSXXXXXX','XXXXXXSS','UUUUUUUU','XXUUUUXX','UUXXXXXX','XXXXXXUU',
                'CXXXXXXX','XCXXXXXX','GXXXXXXX','XACGIOSX','XXXXXXXE','XXXXXXEX','XXXXXXXG','XCEFGSXX']


    sizes = [60]
    fonts = ['arial']
    xshift = [0]
    yshift = [0]

    #create train and val folders
    for m in ['train']:
        for n,f in enumerate(wordlist):
            target_path = path_out+m+'/'+str(n).zfill(2)+'_'+f
            os.makedirs(target_path)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for i,w in enumerate(wordlist):
        gc.collect()
        print (w,)
        path = path_out+'train/'+str(i).zfill(2)+'_'+w+'/'
        n = 0
        for f in fonts:
            for s in sizes:
                for x in xshift:
                    for y in yshift:
                        n = n+1
                        gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

    return 'done'

CreateExpWords()

    #%% Study Position Coding
def CreateExpWords(path_out='stimuli/PosTuning/',num_train=100):

    wordlist = ['SXXX    ', 'XSXX    ', 'XXSX    ', 'XXXS    ',
                ' SXXX   ', ' XSXX   ', ' XXSX   ', ' XXXS   ',
                '  SXXX  ', '  XSXX  ', '  XXSX  ', '  XXXS  ',
                '   SXXX ', '   XSXX ', '   XXSX ', '   XXXS ',
                '    SXXX', '    XSXX', '    XXSX', '    XXXS',
                'S X X X ', 'X S X X ', 'X X S X ', 'X X X S ',
                ' S X X X', ' X S X X', ' X X S X', ' X X X S',
                ]


    sizes = [60]
    fonts = ['Consolas']
    xshift = [0]
    yshift = [0]

    #create train and val folders
    for m in ['train']:
        for n,f in enumerate(wordlist):
            target_path = path_out+m+'/'+str(n).zfill(2)+'_'+f
            os.makedirs(target_path)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for i,w in enumerate(wordlist):
        gc.collect()
        print (w,)
        path = path_out+'train/'+str(i).zfill(2)+'_'+w+'/'
        n = 0
        for f in fonts:
            for s in sizes:
                for x in xshift:
                    for y in yshift:
                        n = n+1
                        gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

    return 'done'

CreateExpWords()

#%% Generating position tuning stimuli for all word selective units
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
    temp = list('        ')
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

temp = list('        ')
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


#%% case invariant stimuli
# def CreateExpWords(path_out='stimuli/CaseInvariance_rnd/',num_train=100):
def CreateExpWords(path_out='stimuli/Casevariance_letters/',num_train=100):
    
    letters = ['a','b','d','e','g','h','n','q','r']
    # letters = ['c','k','o','p','s','u','v','x','z']
    
    words = []
    for i in range(250):
        words.append(''.join(np.array(letters)[np.random.randint(0,9,6)]))
    
    # wordlist = words
    wordlist = letters

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
        for indx in range(2):
            path = path_out+'train/'+str(n).zfill(4)+'_'+w+'/'
            gen2(savepath=path, text=w, index=indx, fontname=fonts[0], size=40, upper=indx)
        n +=1

    return 'done'

CreateExpWords()

#%% Bigram stimuli for dissimilarity analysis

def CreateExpWords(path_out='stimuli/bigrams/'):
    
    letters = ['E','A','S','T','I','R','N']
    # letters = ['a','b','d','e','g','h','n']

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

#%%  FMRI stim
def CreateExpWords(path_out='stimuli/fmri_stim/'):

    wordlist = ['OXX   ', 'XOX   ', 'XXO   ',
				' OXX  ', ' XOX  ', ' XXO  ',
				'  OXX ', '  XOX ', '  XXO ',
				'   OXX', '   XOX', '   XXO',
				'TBB   ', 'BTB   ', 'BBT   ',
				' TBB  ', ' BTB  ', ' BBT  ',
				'  TBB ', '  BTB ', '  BBT ',
				'   TBB', '   BTB', '   BBT',
				'EQQ   ', 'QEQ   ', 'QQE   ',
				' EQQ  ', ' QEQ  ', ' QQE  ',
				'  EQQ ', '  QEQ ', '  QQE ',
				'   EQQ', '   QEQ', '   QQE',
                ]


    sizes = [60]
    fonts = ['Consolas']
    xshift = [0]
    yshift = [0]

    #create train and val folders
    for m in ['train']:
        for n,f in enumerate(wordlist):
            target_path = path_out+m+'/'+str(n).zfill(2)
            os.makedirs(target_path)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for i,w in enumerate(wordlist):
        gc.collect()
        print (w,)
        path = path_out+'train/'+str(i).zfill(2)+'/'
        n = 0
        for f in fonts:
            for s in sizes:
                for x in xshift:
                    for y in yshift:
                        n = n+1
                        gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=1)

    return 'done'

CreateExpWords()
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import numpy as np

words = np.load('Ewordlist.npy')
    
    
####
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
    
    img.save(savepath+text+str(index)+'.jpg')

    
######################
path_out='wordsets_EN2/'
num_train=1300
num_val=50

#define words, sizes, fonts
wordlist = words
sizes = [30, 35, 40, 45, 50, 55, 60, 65, 70]
fonts_tr = ['arial','times']
fonts_val = ['comic','cour','calibri']

xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
yshift = [-30, -15, 0, 15, 30]

#create train and val folders 
for m in ['train', 'val']:
    for f in wordlist:
        target_path = path_out+m+'/'+f
        os.makedirs(target_path, exist_ok= True)

def process(w):
    print (w,)
    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for n in range(num_train + num_val):
        if n < num_train:
            path = path_out+'train/'+w+'/'
            fonts = fonts_tr
        else:
            path = path_out+'val/'+w+'/'
            fonts = fonts_val
        
        f = random.choice(fonts)
        s = random.choice(sizes)
        u = random.choice([0,1])
        x = random.choice(xshift)
        y = random.choice(yshift)
        gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)

results = Parallel(n_jobs=6)(delayed(process)(i) for i in wordlist)

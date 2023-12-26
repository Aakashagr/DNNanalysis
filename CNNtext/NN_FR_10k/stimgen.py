from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, random, gc
from tqdm import tqdm
import numpy as np
import unidecode

words = []
f = open("wordlist.txt", "r")
lines = f.readlines()

for l in lines:
    words.append(unidecode.unidecode(l).strip())
words = np.unique(words)
words = np.array(words)[np.ceil(np.arange(0, len(words), len(words)/10000)).astype(int)]   

    
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
def CreateWordSet(path_out='wordsets_FR/',num_train=140, num_val=5):
    #define words, sizes, fonts
    wordlist = words
    sizes = [40, 50, 60, 70, 80]
    fonts_tr = ['arial','times']
    fonts_val = ['comic','cour','calibri']

    xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    yshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    
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

    return 'done'


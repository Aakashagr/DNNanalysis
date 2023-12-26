# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:44:59 2022

@author: Aakash
"""

import numpy as np
import pandas as pd
import string
from collections import Counter
pd.set_option('display.max_colwidth', None)

#%%
# data = pd.read_csv('telugu_corpus_2012_03_30/Tel_Blogs.txt', delimiter = "\t", header=None)
data = pd.read_csv('Malayalam_Corpus_2011_04_08/Ml_Blogs.txt', delimiter = "\t", header=None)
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~¨«¿“”„‘’₨₹1234567890–'
data.iloc[:,4] = data.iloc[:,4].str.replace('[{}]'.format(string.punctuation), '')
corpus = data.iloc[:30000,4].to_string(index = False)
words = np.array(corpus.split())

#%%

counter = Counter(words)
wordlist = counter.most_common(1200)

wl = []
for i in range(len(wordlist)):
    qt = str(wordlist[i][0])
    if (len(qt) < 10):
        wl.append(qt)
# df = pd.DataFrame(wl)
# df.to_csv('Telugu_words.csv', index=False, encoding='utf-8')
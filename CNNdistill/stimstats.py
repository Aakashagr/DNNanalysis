#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:10:49 2022

@author: aakash
"""

fig, axs = plt.subplots(1,2, figsize=(8,10), facecolor='w', edgecolor='k')
axs = axs.ravel();

charcoef = np.reshape(np.mean(Xmat,0),[26,8])
maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval*1.2
for r in range(np.size(charcoef,0)):
#         strchar = string.ascii_lowercase[r]
    strchar = string.ascii_uppercase[r]
    for c in range(np.size(charcoef,1)):
        strcol = 'red' if charcoef[r,c] >0 else 'blue'
        axs[i].text( c+.5,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
        axs[i].set_xticks(np.arange(0.5,9,1)); axs[i].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
        axs[i].set_yticks(np.arange(0.5,27,1)); axs[i].set_yticklabels([]);
        axs[i].yaxis.set_ticks_position('none')
    


#%%
charcoef = np.reshape(np.mean(Xmat,0),[26,8])
plt.plot(np.median(charcoef,1))
plt.xticks(range(26),string.ascii_uppercase)
plt.title('Median coefficients across position')

plt.figure()
plt.plot(np.std(charcoef,1))
plt.xticks(range(26),string.ascii_uppercase)
plt.title('std dev. coefficients across position')


plt.figure()
plt.plot(np.sum(charcoef,1))
plt.xticks(range(26),string.ascii_uppercase)
plt.title('Sum coefficients across position')
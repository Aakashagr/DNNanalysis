#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:38:18 2022

@author: aakash
"""

os.makedirs('plots/postuning_h_rf/', exist_ok=True)

for bias in np.arange(0,890,10):
    folderid = 'WSunit_' + str(bias).zfill(3)

    max_len = max(map(len, stimword));
    fig, axs = plt.subplots(1,10, figsize=(40,10), facecolor='w', edgecolor='k')
    axs = axs.ravel();
    
    for i in np.arange(bias,10+bias):
        # print(val)
        # Visualizing the coefficients
        charcoef = np.reshape(coefMat[i,:],[26,8])
        maxval = np.max(abs(charcoef)); charcoef = charcoef*25/maxval
        for r in range(np.size(charcoef,0)):
    #         strchar = string.ascii_lowercase[r]
            strchar = string.ascii_uppercase[r]
            for c in range(np.size(charcoef,1)):
                strcol = 'red' if charcoef[r,c] >0 else 'blue'
                axs[i-bias].text( c,25-r, strchar, fontsize = abs(charcoef[r,c]), color = strcol)
                axs[i-bias].set_xticks(np.arange(0.5,9,1)); axs[i-bias].set_xticklabels(['1','2','3','4','5','6','7','8',''], fontsize = 16);
                axs[i-bias].set_yticks(np.arange(0.5,27,1)); axs[i-bias].set_yticklabels([]);
                axs[i-bias].yaxis.set_ticks_position('none')

    
        axs[i-bias].set_title('unit #: ' + str(wordSelUnit[i])+ ': r2 = '+str(round(rfit[i],2)), fontsize = 16)
        
    fig.savefig('plots/postuning_h_rf/' +folderid + '.png')
    plt.close(fig)
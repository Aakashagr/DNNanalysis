# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:01:46 2023

@author: Aakash
"""


    # "for i,layer in enumerate(ltype):\n",
    # "    tr_pattern = np.array(nBli[layer])[train_id,:]\n",
    # "    pcax.fit(tr_pattern)\n",
    # "    pc_tr = pcax.transform(tr_pattern)\n",
    # "    \n",
    # "    pc_tr = np.c_[pc_tr, np.ones((np.shape(pc_tr)[0],1))]\n",
    # "    tr_label = np.identity(248)\n",
    # "    weights = (tr_label@np.linalg.pinv(pc_tr).T).T\n",
    # "\n",
    # "    tr_pred = pc_tr@weights\n",
    # "    train_acc[i] = np.mean(np.argmax(tr_pred,1) == range(248))\n",
    # "\n",
    # "    for t in range(8):\n",
    # "        ts_pattern = np.array(nBli[layer])[test_id[t],:]\n",
    # "        pc_ts = pcax.transform(ts_pattern)\n",
    # "        pc_ts = np.c_[pc_ts , np.ones((np.shape(pc_ts)[0],1))]\n",
    # "        test_pred = pc_ts@weights\n",
    # "        test_acc[i,t] = np.mean(np.argmax(test_pred,1) == range(248))\n",
    # "\n",
    # "    "
	

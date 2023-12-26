#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:16:49 2022

@author: aakash
"""


import pickle
with open('WSunits_lit_it.pkl', 'rb') as f:
 	wordSelUnit = pickle.load(f)


wsdiff = np.diff(wordSelUnit)
qx = np.where(wsdiff> 10)[0]

sel_units = [wordSelUnit[32:47], wordSelUnit[71:84], wordSelUnit[150:163], 
             wordSelUnit[381:392],wordSelUnit[515:530],wordSelUnit[600:610],
             wordSelUnit[708:719],wordSelUnit[816:824]]

sel_units = sum(sel_units,[])

with open('WSunits_lit_subset_it.pkl', 'wb') as f:
 	pickle.dump(sel_units, f)

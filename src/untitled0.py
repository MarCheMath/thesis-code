#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:42:02 2019

@author: marche
"""
import wavelet_basis as wb
import numpy as np

w=wb.generate_basis(im_dim=28,db_num=8).reshape(-1,784).T
print(w)
#Wav={}
#for i in range(1):
#    i=i+1
#    db_num=i
#    W=wb.generate_basis(im_dim=28,db_num=db_num).reshape(-1,784).T
#    w,s = wb.best_singularvalues(W,784)
#    Wav[db_num]=[w,s]
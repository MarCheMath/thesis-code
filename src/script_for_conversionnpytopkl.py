#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
from scipy.io import loadmat as load
import numpy as np
#
#for i in [10,25,50,100,200,300,400,500,600,750]:
#    rec = load("../../Mark_comparison/results/{}/pca/rec_err_{}_total_.mat".format(i,i))['rec_err']
#    rec = {key:rec[0,key] for key in range(rec.shape[1])}
#    with open('../../Mark_comparison/results/{}/pca/l2_losses.pkl'.format(i),'wb') as f:
#        pickle.dump(rec, f)


for i in [300,400,500,600,750]:#[10,25,50,100,200,300,400,500,600,750]:
    rec = np.load("../../Mark_comparison/results/{}/pca/l2_losses.npy".format(i))#['l2_losses']
    rec = {key:rec[0,key] for key in range(rec.shape[1])}
    with open('../../Mark_comparison/results/{}/pca/l2_losses.pkl'.format(i),'wb') as f:
        pickle.dump(rec, f)
        
#for i in [10,25,50,100,200,300,400,500,600,750]:
#    rec = np.load("../../Mark_comparison/results/{}/pca/m.npy".format(i))#['l2_losses']
#    rec = {key:rec[0,key] for key in range(rec.shape[1])}
#    with open('../../Mark_comparison/results/{}/pca/l2_losses.pkl'.format(i),'wb') as f:
#        pickle.dump(rec, f)
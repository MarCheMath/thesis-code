#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:40:17 2019

@author: cheng
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:34:09 2019

@author: marche
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import errno

#sparsity_representation('detail_sparsity',True)

def sparsity_representation(mode,save,refresh=False):
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    img = mnist.test.images
    if mode == 'mean_sparsity':
        mean_sparsity(img,save)
    elif mode == 'detail_sparsity':
        sparsity_plot(img,save,refresh)
    
    
def mean_sparsity(img,save):    
    spars_mean = np.mean(np.count_nonzero(img,axis=1))
    print(spars_mean)
    if save == True:
        path = '../estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        np.save(path+'mean_sparsity.npy',spars_mean)
    return spars_mean

def sparsity_plot(img,save,refresh):
    path = '../estimated/mnist/pure_sparsity/'
    img_dim = len(img[0])
    img_numb = len(img)
    max_it = 784 #img_dim
    it_max = min(img_dim,max_it)
    
    flag, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_euclid = chkpoint(path,it_max,refresh)
    
    sortedArr=np.argsort(img,axis=-1)
    indices = (np.ones((img_numb,img_dim))*np.asarray(range(img_numb))[:,None]).astype(int)
    if flag:
        for k in range(it_max):#speed could be improved
            ind_j = sortedArr[:,:img_dim-k]    
            ind_i = indices[:,:img_dim-k]
            img_k = np.copy(img)
            img_k[ind_i,ind_j]=0 #one can store this
            rep_err[k] = np.mean(np.linalg.norm(img_k-img, axis = 0))
    
            print('Iteration: {}'.format(k))
            print('The mean representation error is: {}'.format(rep_err[k]))
            rep_err_std[k] = np.std(np.linalg.norm(img_k-img, axis = 0))
            print('The stdev of representation error is: {}'.format(rep_err_std[k]))
            rep_err_pixel[k] = np.mean(np.mean((img_k-img)**2, axis = 0))
            print('The mean representation error is: {}'.format(rep_err_pixel[k]))
            rep_err_pixel_euclid[k] = np.mean(np.sqrt(np.sum((img_k-img)**2, axis = 0))/img_dim)
            print('The mean representation error is: {}'.format(rep_err_pixel_euclid[k]))    
       
    plot_error(path, rep_err, save, 'l2', 0)
    plt.figure()
    plot_errorbar(path, rep_err, rep_err_std, save, 'l2', 0)
    plt.show()
    plot_error(path, rep_err_pixel,save,'pixelwise_wo_sqrt',0)
    plot_error(path, rep_err_pixel_euclid,save,'pixelwise_with_sqrt',0)
    if save:        
        np.save(path + 'rep_err.npy', rep_err)
        np.save(path + 'rep_err_std.npy', rep_err_std)
        np.save(path + 'rep_err_pixel.npy', rep_err_pixel)
        np.save(path + 'rep_err_pixel_euclid.npy', rep_err_pixel_euclid)
        print(path + 'rep_err')


def chkpoint(path,it_max,refresh=False):    
    if (not(refresh) and os.path.isfile(path+'rep_err.npy') and os.path.isfile(path+'rep_err_std.npy') and os.path.isfile(path+'rep_err_pixel.npy') and os.path.isfile(path+'rep_err_pixel_euclid.npy')):
        print(os.getcwd())
        rep_err = np.load(path + 'rep_err.npy')
        rep_err_std = np.load(path + 'rep_err_std.npy')
        rep_err_pixel = np.load(path + 'rep_err_pixel.npy')
        rep_err_pixel_euclid = np.load(path + 'rep_err_pixel_euclid.npy')        
        return False, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_euclid
    else:
        rep_err = np.ones(it_max,)
        rep_err_std = np.ones(it_max,)
        rep_err_pixel = np.ones(it_max,)
        rep_err_pixel_euclid = np.ones(it_max,)
        return True, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_euclid

def plot_error(path, data,save,name,lim):
    plt.figure()
    plt.plot(range(len(data)),data)
    if lim!='Auto':
        plt.ylim(bottom=lim)
    if save == True:
        #path = '../estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        plt.savefig(path+'error_plot_sparsity_{}.pdf'.format(name))
        np.save(path+'error_plot_sparsity_{}.npy'.format(name),data)
    plt.show()

def plot_errorbar(path, data,stdv, save,name,lim,ax='',color=None,ecolor=None,indices=[250, 260,270,280,290,300, 400]):
#    print(np.array(data))
    #indices = [5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500, 783]#range(len(data))
    data = np.array(data)[np.array(indices)]
    stdv = np.array(stdv)[np.array(indices)]    
    if ax=='':
        plt.errorbar(indices, data, yerr=1.96*stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
    else:
        ax.errorbar(indices, data, yerr=1.96*stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
    if lim!='Auto':
        plt.ylim(bottom=lim)
    if save == True:
        #path = '../estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        plt.savefig(path+'error_plot_sparsity_{}.pdf'.format(name))
        np.save(path+'error_plot_sparsity_{}.npy'.format(name),data)
    
        
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
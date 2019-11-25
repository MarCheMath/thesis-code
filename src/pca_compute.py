#!/usr/bin/env python2

import os
print(os.system('pwd'))
import numpy as np
from argparse import ArgumentParser
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA #Try out different values
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
import utils_lasso

#def compute_np():
#    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#    X = mnist.train.images.T  
#    n = X.shape[0]
#    pca = PCA(n_components=n)
#    X = StandardScaler().fit_transform(X) #zero meaning AND UNIT VARIANCE
#    x_mean = np.mean(X,axis=1)
#    print x_mean
#    principalComponents = pca.fit_transform(X)
##    U,_ = np.linalg.svd(X.dot(X.T))
#    np.save('../pca_basis_normalized.npy',principalComponents)
#    np.save('../pca_mean_normalized.npy',x_mean)

def get_X():
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    X = mnist.train.images.T   
    return X
    
def compute_pca(X,normalization,mode): #Own solution
    print('mode: {}, normalization: {}'.format(mode,normalization))
    num_samples = X.shape[1]
    if mode == 'own':
        x_mean = np.mean(X,axis=1)
        if normalization == 'normalized':
            Xm = (X-x_mean[:,np.newaxis])
            Xv = np.sqrt(np.var(X,axis=1)[:,np.newaxis])
            Xv[Xv==0]=1 #meaned value zero, so enumerator can be set freely
            X = Xm/Xv
        X_var = np.cov(X) #empirical covariance (unbiased estimator)
        pca_basis,_,_ = np.linalg.svd(X_var)
    elif 'sklearn' in mode:
        X=X.T #expects X.shape = [n_sample,n_feature]
        if 'Sparse' in mode:
            pca_obj = SparsePCA
            print 'sparse'
        elif 'DictionaryLearning' in mode:
            pca_obj = DictionaryLearning
        else:
            pca_obj = PCA
        n = X.shape[1]              
        if normalization == 'normalized':
            if 'Sparse' in mode:
                pca = pca_obj(n_components=n,normalize_components=True)
            elif 'DictionaryLearning' in mode:
                dim = utils_lasso.get_key('DictionaryLearning',mode,int)
                pca = pca_obj(n_components=dim)
            else:
                #zero meaning AND UNIT VARIANCE
                X = StandardScaler().fit_transform(X)
                pca = pca_obj(n_components=n)                
        elif normalization == 'whiten':
            pca = pca_obj(n_components=n,whiten=True) #rescales components to unit variance
        else:
            if 'Sparse' in mode:
                pca = pca_obj(n_components=n,normalize_components=False)
            elif 'DictionaryLearning' in mode:
                dim = utils_lasso.get_key('DictionaryLearning',mode,int)
                pca = pca_obj(n_components=dim)
            else:
                pca = pca_obj(n_components=n,whiten=False)
        x_mean = np.mean(X,axis=0).T
        pca_basis = pca.fit(X).components_.T
    else:
        raise NotImplementedError("Mode is {}, but only 'sklearn' and 'own' are implemented".format(mode))
    np.save('./pca_basis_{}_{}_samples_{}.npy'.format(normalization,mode,num_samples),pca_basis)
    np.save('./pca_mean_{}_{}_samples_{}.npy'.format(normalization,mode,num_samples),x_mean)
    print('Finished calculation')

if __name__ == '__main__':
    PARSER = ArgumentParser()

#    PARSER.add_argument('--submit-mode', type=str, default='tmux', help='Selected process mode')
    PARSER.add_argument('--training-size', type=int, default=55000, help='used sample size')
    PARSER.add_argument('--normalization', type=str, default='unnormalized', help='hidden dimension n_z')
    PARSER.add_argument('--mode', type=str, default='sklearnDictionary', help='hidden dimension n_z')
                                                                                                              
    HPARAMS = PARSER.parse_args()

    X=get_X()    
    X = X[:,:HPARAMS.training_size]
    compute_pca(X,HPARAMS.normalization,HPARAMS.mode)

#compute_pca(X,'normalized','sklearnDictionaryLearning')
#compute_pca(X,'normalized','sklearnSparse') #not very meaningful as it requiers the dictionary itself to be sparse
#compute_pca(X,'unnormalized','sklearnSparse') #like above
#compute_pca(X,'normalized','own')
#compute_pca(X,'unnormalized','own')
#compute_pca(X,'normalized','sklearn')
#compute_pca(X,'standardScaler','sklearn')
#compute_pca(X,'unnormalized','sklearn')
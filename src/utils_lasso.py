#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:12:40 2019

@author: marche
"""

import numpy as np
import cvxopt
from cvxopt import matrix
import os
import sys
import re
import cvxpy as cp
import warnings

from matplotlib import pyplot as plt


def get_y(mode,y_batch_val,A):
    if 'affine' in mode: #affine in PCA
        x_mean = np.load('./pca_mean.npy')
        y = y_batch_val - A.dot(x_mean) #normalization
    else:
        y = y_batch_val
    return y.T

def get_x(mode,x):
    if 'affine' in mode: #affine in PCA
        x_mean = np.load('./pca_mean.npy')
        return x+x_mean
    else:
        return x

    

def load_W(hparams, A_val,y_val):
    mode = hparams.tv_or_lasso_mode
    if hparams.model_types[0] == 'omp':
        in_type = mode
    else:
        in_type = hparams.model_types[0]
    if 'lasso-wavelet' in in_type:
        re1 = re.match('.*-(.*)-([0-9]*)$',mode) 
        redundancy = [x.group(1) for x in [re1] if x != None][0]
        pix = [x.group(2) for x in [re1] if x != None]
        pix = [int(x) if x !='' else 1 for x in pix][0]
        addendum=''
        if hparams.wavelet_type !='':
            addendum='-'+hparams.wavelet_type
        W = np.load('./{}wavelet_basis-{}-{}{}.npy'.format(hparams.matlab,redundancy,pix,addendum)).reshape(-1,784)
        W = np.asarray(W).T
    elif 'lasso-pca' in in_type:    
        normalization = get_key('',mode,'normalized|unnormalized|standardScaler')
        pca_computation = get_key('',mode,'own|sklearn|sklearnDictionaryLearning|sklearnSparse|rawDict')
        samples = get_key('samples',mode,int,standard=0)
        learndim = get_key('DictionaryLearning',mode,int,standard=0)
        praependix = './pca_basis_{}_{}'.format(normalization,pca_computation)
        appendix = ''
        if learndim != 0:
            appendix = appendix + '{}'.format(learndim) 
        if samples != 0:
            appendix = appendix + '_samples_{}'.format(samples) 
        print praependix+appendix+'.npy'
        W = np.load(praependix+appendix+'.npy')
        W = W[:,:int(W.shape[1]*get_key('pcafraction',mode,float))]
        print int(W.shape[1]*get_key('pcafraction',mode,float))
    elif 'tv-norm' in in_type:
        if A_val is None:
            dim_n = y_val.shape[0]
        else:
            dim_n = A_val.shape[1]
        gradX,gradY = constr_grad(int(np.sqrt(dim_n)))
        W = np.concatenate((gradX,gradY))
    elif 'lasso' in in_type: #position important
        if A_val is None:
            dim_n = y_val.shape[0]
        else:
            dim_n = A_val.shape[1]
        W = np.eye(int(dim_n))   #sqrt? 
    elif 'dict' in in_type:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
        rawDictLength = get_key('rawDictLength',hparams.tv_or_lasso_mode,int,standard=784)
        W = mnist.train.images[:,:rawDictLength]
#-----------TODO alternatively do W = None as special case TODO---------------
    return W


def check_bestkTermApprox(mode, W, x, treat_as_if_ONB = True):#rows of W are phi_i
    if treat_as_if_ONB:
        W = W[:784,:784].T
    if W.shape[0] == W.shape[1]:
        if np.linalg.norm(W.dot(W.T)-np.eye(W.shape[0]))/(W.shape[0]**2) <=0.01: #check wether ONB
            if 'synthesis' in mode:
                Wx = W.dot(x)
                ind = np.argsort(Wx)[:get_key('kterm',mode,int)]
                xhat = W.T[:,ind].dot(Wx[ind])
                return xhat
            raise ValueError("Best k term approx can only be done in synthesis mode, but mode {} does not contain synthesis".format(mode))
        warnings.warn("W should be an ONB, but {} is not orthogonal or not normed!".format(W))
    warnings.warn("W should be an ONB, but {} is not quadratic (has shape {})!".format(W,W.shape))
    print('Starting brute force approach!')
    kterm = get_key('kterm',mode,int)
    import itertools
    x_dict = {}
    val_dict = {}
    for k in range(kterm):
        x_dict.update({i:W[:,list(i)].dot(np.linalg.solve(W[:,list(i)]),x) for i in itertools.combinations(range(W.shape[1]),k)})
        val_dict.update({i:np.linalg.norm(xhat-x) for i,xhat in x_dict.iteritems()})
    return x_dict[min(val_dict, key=val_dict.get)]
                  
                
def brute_force(mode, A,W,y):
    if 'synthesis' in mode:
        kterm = get_key('kterm',mode,int)
        import itertools
        x_dict = {}
        val_dict = {}
        for k in range(kterm):
            k=k+1
            x_dict.update({i:W[:,list(i)].dot(np.linalg.lstsq(A.dot(W[:,list(i)]),y)[0]) for i in itertools.combinations(range(W.shape[1]),k)})
            val_dict.update({i:np.linalg.norm(A.dot(x)-y) for i,x in x_dict.iteritems()})
        return x_dict[min(val_dict, key=val_dict.get)]
    else:
        raise NotImplementedError('Not implemented yet')

def cvxpy_problem(mode,W,A,AW,y,weight):
    sample_n = y.shape[1]
    eps = 0.001
    if 'synthesis' in mode:
        dim_v = W.shape[1]
        Psi = np.eye(dim_v)
        Phi = AW
        dim_xhat = W.shape[0]
    elif 'analysis' in mode:
        dim_v = W.shape[0]
        Psi = W.T
        Phi = A  
        dim_xhat = A.shape[1]
    else:
        dim_v = W.shape[1]
        Psi = W
        Phi = A
        dim_xhat = A.shape[1]
        
    x_hat_batch = np.zeros((sample_n,dim_xhat))
    variable = np.zeros((dim_v,sample_n))
    
    if dim_v > 100: #If dimension too large, process each image seperately.
        repeat = sample_n
        sample_n = 1
    else:
        repeat = 1
    y_batch = y
    weight_batch = weight
        
    for batch_it in range(repeat):
        y = y_batch[:,batch_it*sample_n:(batch_it+1)*sample_n]
        if weight_batch.size != 0:
            weight = weight_batch[:,batch_it*sample_n:(batch_it+1)*sample_n]        
        def _cvxpy_batch_(mode,error_solving = False): #closure function to keep environment variables, but recall it in case constrained problem is infeasible and mode has to be changed to reg.
            var,objective = cvxpy_get_var_and_objective(mode,dim_v,sample_n,weight,Phi,Psi,y)
            constraints = cvxpy_get_constraint(mode, Phi,Psi,y,var,W)        
            prob = cp.Problem(objective, constraints)
            try:
                if error_solving:
                    prob.solve(verbose=error_solving, eps = 1e-2) #assumend solver is SCS
                else:
                    prob.solve(verbose=error_solving)        
            except cp.error.SolverError:
                warnings.warn('QP failed to converge, set tolerance up.')
                prob.solve(verbose=True,eps_abs = 0.01) #assumed solver is OQSP    
            if prob.status == 'infeasible':
                if 'reg' not in mode:
                    warnings.warn('The (In)Equation system is infeasible. Assuming W has not full rank (if full rank, this is an error!). Restarting problem by changing constraints into regularizers')
                    mode_tmp = mode.replace('constrInEq','reg')
                    mode_tmp = mode_tmp.replace('constrEq','reg')
                    _cvxpy_batch_(mode_tmp,error_solving = True)
                    return
                else:
                    raise ValueError("The QP is infeasible, although the mode is 'regularizing'. This can not be.")
            if 'synthesis' in mode:
        #            print var.value
        #            print np.sum(var.value>0.001)
        #            plt.figure()
        #            plt.plot(var.value)
        #            plt.show()
                x_hat = np.asarray(W.dot(var.value).T)
            elif 'analysis' in mode:
                x_hat = np.asarray(var.value.T)
            else:
                x_hat = np.asarray(var.value.T)    
            variable[:,batch_it*sample_n:(batch_it+1)*sample_n] = var.value
            x_hat_batch[batch_it*sample_n:(batch_it+1)*sample_n,:] = x_hat
            print('Processed {} percent of the data'.format(float(batch_it+1)/repeat*100))    
        _cvxpy_batch_(mode)
            
    weight_batch = 1/(np.abs(Psi.dot(variable))+eps)
    return (x_hat_batch,weight_batch) 

def cvxpy_get_constraint(mode, Phi,Psi,y,var,W):
    if 'constrEq' in mode:
        constraints = [Phi*var == y]
    elif 'constrInEq' in mode:
        alpha = get_key('alpha',mode,float)
        constraints = [cp.norm(Phi*var - y) <= alpha]
    elif 'reg' in mode:
        constraints = []
    elif 'bestkTermApprox' in mode:
        k = get_key('kterm',mode,int)
        constraints = [cp.norm(Psi*var,1) <= k]
    else:
        raise NotImplementedError 
    if 'nonneg' in mode:
        if 'synthesis' in mode:
            constraints.append(W*var>=0)
        else:
            constraints.append(var>=0)
    return constraints

def cvxpy_get_var_and_objective(mode,dim_v,sample_n,weight,Phi,Psi,y):
    if 'reg' in mode:
        alpha = get_key('alpha',mode,float)
        try: #version of cvxpy (no downward compatibility)
            var = cp.Variable(dim_v,sample_n)
            if weight.size != 0:
                objective = cp.Minimize( alpha * cp.norm(Phi*var-y) + cp.sum_entries(cp.mul_elemwise(weight,cp.abs(Psi*var))))
            else:
                objective = cp.Minimize( alpha * cp.norm(Phi*var-y) + cp.norm(Psi*var,1))
        except:
            var = cp.Variable((dim_v,sample_n))
            if weight.size != 0:
                objective = cp.Minimize( alpha * cp.norm(Phi*var-y) + cp.sum(cp.multiply(weight,cp.abs(Psi*var))))    
            else:
                objective = cp.Minimize( alpha * cp.norm(Phi*var-y) + cp.norm(Psi*var,1))      
    elif 'bestkTermApprox' in mode:
        x = y
        objective = cp.Minimize(cp.norm(Psi*var-x)) 
    else:
        try: #version of cvxpy (no downward compatibility)
            var = cp.Variable(dim_v,sample_n)
            if weight.size != 0:
                objective = cp.Minimize( cp.sum_entries(cp.mul_elemwise(weight,cp.abs(Psi*var))))
            else:
                objective = cp.Minimize(cp.norm(Psi*var,1))
        except:
            var = cp.Variable((dim_v,sample_n))
            if weight.size != 0:
                objective = cp.Minimize(cp.sum(cp.multiply(weight,cp.abs(Psi*var))))    
            else:
                objective = cp.Minimize(cp.norm(Psi*var,1)) 
    return (var,objective)
                        
def tridiag(n,a,b,c):
    M1 = np.diag(np.ones(n)*b)
    M2 = np.zeros((n,n))
    M2[:-1,1:] = np.diag(np.ones(n-1)*a)
    M3 = np.zeros((n,n))
    M3[1:,:-1] = np.diag(np.ones(n-1)*c)
    return M1+M2+M3

def constr_grad(n,mode='diff1_periodic'):
    print('gradient created according to {}'.format(mode))
#    n_dim = n**2
    if 'np_gradient' in mode:
        raw_grad = np.array(np.gradient(np.eye(n)))[0]
    elif 'diff1' in mode:
        raw_grad = tridiag(n,-1,1,0)
#        raw_grad = np.delete(raw_grad, -1, axis=0) #no periodicity
        if 'periodic' in mode:
            raw_grad[-1,0] = -1
        elif 'whole' in mode :
            raw_grad[-1,0:-1:2] = 1
            raw_grad[-1,1:-1:2] = -1
        elif 'raw' in mode:
            pass
        else:
            np.delete(raw_grad,-1,0)
    elif 'diff2' in mode:
        raw_grad = tridiag(n,-1,2,-1)
        if 'periodic' in mode:
            raw_grad[-1,0] = -1
        elif 'whole' in mode :
            raw_grad[-1,0:-1:2] = 1
            raw_grad[-1,1:-1:2] = -1
        elif 'raw' in mode:
            pass
        else:
            np.delete(raw_grad,-1,0)
#        raw_grad = np.delete(raw_grad, -1, axis=0) #no periodicity
    identity = np.eye(n)
    gradX = np.kron(identity,raw_grad)
    gradY = np.kron(raw_grad, identity)
    return (gradX,gradY)
    
def tv_cvxopt_partial(A_valT, y_batch_val,hparams):
    mode = hparams.tv_or_lasso_mode
    if 'reg' in mode:
        alpha = get_key('alpha',mode,float)
    maxit = get_key('reweight',mode,int)
    
    eps = 0.001
    
    A_val = A_valT.T
    W = load_W(hparams, A_val,y_batch_val)    
    if 'synthesis' in mode:
        Psi = np.eye(W.shape[1])
        Phi = A_val.dot(W)
    elif 'analysis' in mode:
        Psi = W.T
        Phi = A_val  
    else:
        Psi = W
        Phi = A_val
    
   
    m_meas = A_val.shape[0]

    dim_w = m_meas
    dim_x = Psi.shape[1]
    dim_u = Psi.shape[0]#2*n_dim
    all_dim = dim_w+dim_x+dim_u
    
    weight = np.ones(dim_u)
    
    for it in range(maxit):        
        if 'reg' in mode:
            P = np.zeros((all_dim,all_dim))
            q = np.zeros(all_dim)
            G = np.zeros((2*dim_u,all_dim))
            A = np.zeros((m_meas,all_dim))
            P[:dim_w,:dim_w] = 2*np.eye(dim_w) #2 because 1/2 factor x^TPx
            q[dim_w:dim_w+dim_u] = alpha * weight       
            G[:dim_u,dim_w:dim_w+dim_u] = -np.eye(dim_u)
            G[:dim_u,dim_w+dim_u:all_dim] = -Psi
            G[dim_u:2*dim_u,dim_w:dim_w+dim_u] = -np.eye(dim_u)
            G[dim_u:2*dim_u,dim_w+dim_u:all_dim] = Psi
            
            A[:,dim_w+dim_u:] = Phi
            A[:,:dim_w] = -np.eye(m_meas)#w
            b = y_batch_val
            sys.stdout = open(os.devnull, 'w')
            solv = cvxopt.solvers.qp(matrix(P),matrix(q),matrix(G),matrix(np.zeros(2*dim_u)),matrix(A),matrix(b))
            sys.stdout = sys.__stdout__
            xhat = np.array(solv.get('x'))[dim_w+dim_u:]
            xhat = np.squeeze(xhat)
        elif 'constrEq' in mode:
            P = np.zeros((dim_x+dim_u,dim_x+dim_u))
            q = np.zeros(dim_x+dim_u)
            G = np.zeros((2*dim_u,dim_x+dim_u))
            A = np.zeros((dim_w,dim_u+dim_x))
            q[:dim_u]= weight       
            
            G[:dim_u,:dim_u] = -np.eye(dim_u)
            G[:dim_u,dim_u:] = -Psi
            G[dim_u:,:dim_u] = -np.eye(dim_u)
            G[dim_u:,dim_u:] = Psi
            
            A[:,dim_u:] = Phi
            b = y_batch_val
            sys.stdout = open(os.devnull, 'w')
            solv = cvxopt.solvers.qp(matrix(P),matrix(q),matrix(G),matrix(np.zeros(2*dim_u)),matrix(A),matrix(b))
            sys.stdout = sys.__stdout__
            xhat = np.array(solv.get('x'))[dim_u:]
            xhat = np.squeeze(xhat)
        else:
            raise NotImplementedError('For cvxopt, only the modes reg and constrEq are allowed! You have {} as mode.'.format(mode))
        weight = 1/(np.abs(Psi.dot(xhat))+eps)
    if 'synthesis' in mode:
        xhat = np.asarray(W.dot(xhat))
    elif 'analysis' in mode:
        xhat = np.asarray(xhat)
    else:
        xhat = np.asarray(xhat)  
    return xhat


#def custom_FISTA(h)

#-----------------------------------------------------------------------------
#Input
#    key:        Signal word, after which the value should be
#    mode:       The string to match
#    type_var:   If int or float, the output gets the specified type. For strings,
#                enter the pattern for the regular expression as string
#                (e.g.: 'normalized|unnormalized'), the output will be a string, too. 
#                In case of multiple matches, the longest/highest will be taken.     
#Output:
#    value:     The searched value
#-----------------------------------------------------------------------------
def get_key(key,mode, type_var,standard = False):
    value_l = _get_key_(key,mode,type_var) #get list of matches (sorted)
    if len(value_l)>0: 
        value = value_l[0]  #if matches, take longest/highest
    else:
        if standard == False:
            if type_var == int or type_var == float:
                value = 1           #if no match, set to standard value (i.e. 1)
            else:
                value = ''
                warnings.warn("You didn't set mode {} to anything matching {}. Setting to '' string.".format(mode,type_var))
        else:
            value = standard
    return value

def _get_key_(key,mode,type_var):
    if type_var == int:
        reg_ex = key+'[^\.][0-9]+[^\.]' #matches last character of key, if key not empty
        if '~' in mode:
            raise ValueError("The character ~ is forbidden!")
        mode = '~'+mode+'~' #assumption: no ~ in mode
    elif type_var == float:
        reg_ex = key+'\d*\.\d+'
    else:
        reg_ex = key+type_var
        type_var = str
    value_l = [val[len(key):] for pat in reg_ex.split('|') for val in re.findall(pat,mode)]
#    re1 = re.match('.*{}({}).*$'.format(key,reg_ex),mode) #abcd, a: ignored chars, b: keyword, c: float or int, d: ignored chars 
#    value_l = [x.group(1) for x in [re1] if x != None]
#    value_l = [type_var(x) if x !='' else 1 for x in value_l]
    if type_var == int:
        if key != '':
            value_l = [x[:-1] for x in value_l]
        else:
            value_l = [x[1:-1] for x in value_l]
        mode = mode[1:-1]
    if type_var == float:
        value_l = value_l+_get_key_(key,mode,int)
    value_l = [type_var(x) for x in value_l if x !='']   
    if type_var == str:
        value_l.sort(key=len,reverse=True)
    else:
        value_l.sort(reverse=True)
    return value_l

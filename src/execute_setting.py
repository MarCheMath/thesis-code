#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
import time
import itertools

def main(hparams):
    base = [
    " --num-input-images 64",
    " --batch-size 64",

    " --mloss1_weight 0.0",
    " --mloss2_weight 1.0",
    " --dloss1_weight 0",
    " --dloss2_weight 0.0",
    " --lmbd 0", 
    " --sparsity 0", 

    " --optimizer-type adam",
    " --momentum 0.9",


    " --save-images",
    " --save-stats",
    " --print-stats",
    " --checkpoint-iter 1",
    " --image-matrix 0",
            ]
    submit_mode = hparams.submit_mode
    qsub_time = hparams.qsub_time
    del hparams.submit_mode
    del hparams.qsub_time
    
    fields = {field:getattr(hparams,field) for field in dir(hparams) if not field.startswith('_')}
    fields = {v:field if type(field)==list else [field] for v,field in fields.iteritems() }
    b = ''.join(base)
    for setting in itertools.product(*fields.values()):
        setting = dict(zip(fields.keys(),setting))
        head = ''.join(['--'+str(v1).replace('_','-')+' ' +str(v2)+' ' for v1,v2 in setting.iteritems()])
        #head=head.replace('_','-')
        if submit_mode == 'qsub':
            head = head.replace(" '",""" "'""")
            head = head.replace("' ","""'" """)
            print(head)
        string = './src/compressed_sensing_mod.py'+ b+' '+head
        ex_string = 'python -u '+string
        print(submit_mode)
        if submit_mode == 'tmux':
            print('tmux new-session -d '+ex_string)
            os.system('tmux new-session -d '+ex_string)
        elif submit_mode == 'qsub':
#            print("qsub -cwd -N 'CS_VAE' -j y -l h_rt=7200 "+string)
#            os.system("qsub -cwd -N 'CS_VAE' -j y -l h_rt=7200 "+string)
            print("qsub -cwd -N 'CS_VAE' -j y -l h_rt={} ".format(qsub_time)+string)
            os.system("echo "+string+ " | qsub -cwd -N 'CS_VAE' -j y -l h_rt={}".format(qsub_time))
        elif submit_mode == 'cluster':
            print("Cluster "+ex_string)
            os.system("Cluster "+ex_string)
        elif submit_mode == 'vanilla':
            print(ex_string)
            os.system(ex_string)            
        else:
            raise NotImplementedError
        #time.sleep(3)#For batch systems, which are not well configured
#        print(string)
#        os.system(string)

  

if __name__ == '__main__':
    PARSER = ArgumentParser()

    PARSER.add_argument('--submit-mode', type=str, default='tmux', help='Selected process mode')
    PARSER.add_argument('--n-z', type=int, nargs = '+', default=-1, help='hidden dimension n_z')
    PARSER.add_argument('--zprior-weight', type=float, nargs = '+', default=0, help='hidden dimension n_z')
    PARSER.add_argument('--stdv', type=float, nargs = '+', default=1, help='hidden dimension n_z')
    PARSER.add_argument('--mean', type=float, nargs = '+', default=0, help='hidden dimension n_z')
    PARSER.add_argument('--max-update-iter', type=int, nargs = '+', default=1000, help='hidden dimension n_z')
    PARSER.add_argument('--num-measurements', type=int, nargs = '+', default=100, help='hidden dimension n_z')    
    PARSER.add_argument('--measurement-type', type=str, nargs = '+', default="'gaussian'", help='hidden dimension n_z')   
    PARSER.add_argument('--model-types', type=str, nargs = '+', default="'vae'", help='hidden dimension n_z')   
    PARSER.add_argument('--num-random-restarts', type=int, nargs = '+', default=10, help='hidden dimension n_z')   
    PARSER.add_argument('--pretrained-model-dir', type=str, nargs = '+', default='./mnist_vae/models/mnist-vae/mnist-vae-flex-100/', help='directory to pretrained model')   
    PARSER.add_argument('--grid', type=str, nargs = '+', default="NoGrid", help='directory to pretrained model')   
    PARSER.add_argument('--eps', type=float, default=0.01, nargs ='+', help='eps for measurement for flex vae (weighted with norm of A)')
    PARSER.add_argument('--qsub-time', type=int, default=50000, help='Time for qsub')
    PARSER.add_argument('--tol', type=int, default=5, help='tolerance for binary search in vae flex')
    PARSER.add_argument('--init-mode', type=str, default='random', help='mode for the initialization in estimator')
    PARSER.add_argument('--flex-chosen', type=str, nargs = '+',default='flexible', help='fixed dimension of the VAE flex (e.g. good for projection)')
    PARSER.add_argument('--use-gpu', action='store_true', help='Whether to use GPUs')
    PARSER.add_argument('--lmbd', type=float, default=0.0, help='Whether to use GPUs')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')
    PARSER.add_argument('--tv-or-lasso-mode', type=str, default='nada', nargs = '+', help='cvxopt-constr, cvxopt-reg,cvxopt-reg-param10, fista')
    PARSER.add_argument('--batch-size', type=int, default=64, help='Batch size of images')
    PARSER.add_argument('--reproducible', type=str, default='None', help='Whether the measurement matrix A is drawn with fixed seed')
    PARSER.add_argument('--omp-k', type=int, default=300, help='Orthogonal Matching Pursuit sparsity parameter')
    PARSER.add_argument('--noise-std', type=float, default=0.0, nargs = '+', help='std dev of noise')
    PARSER.add_argument('--kterm', type=int, default=-1, nargs = '+', help='For representation system to make incomplete')
    PARSER.add_argument('--input-type', type=str, default='full-input', nargs = '+', help='input type')
    PARSER.add_argument('--dataset', type=str, default='mnist', nargs = '+', help='Dataset to use')
    PARSER.add_argument('--emd-bol', type=str, default = 'True', help='emd loss logged')
    PARSER.add_argument('--tolerance-checking', type=str, default='non-squared', nargs = '+', help='Tolerance checking w.r.t. euclidian norm or squared euclidian norm')
    PARSER.add_argument('--strict-checking', type=str, default='strict', nargs = '+', help='When using alternating checking, use only the grid points')
    PARSER.add_argument('--repetition-bol', type=str, default = 'False', nargs = '+', help='Whether to repeat generator training as many times as grid points')
    PARSER.add_argument('--fair-counter', type=str, default='unequal', help='If and how many times the fixed version is reiterated to make up for the additional optimization')
    PARSER.add_argument('--input-seed', type=str, default='no_seed', help='For random-test input mode fixes a seed')
    PARSER.add_argument('--fair-counter-end', type=int, default=1, help='If and how many times the final iteration is reiterated to improve the optimization')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    PARSER.add_argument('--wavelet-type', type=str, default='db1selected',nargs = '+', help='Which wavelet type to use')
    PARSER.add_argument('--matlab', type=str, default='nada', help='Wavelet case: Should use python generated or matlab generated wavelet systems')
    PARSER.add_argument('--class-bol', type=str, default = 'True', help='emd loss logged')

                                                                    
    HPARAMS = PARSER.parse_args()

    main(HPARAMS)
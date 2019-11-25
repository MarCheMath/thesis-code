"""Create and save the wavelet basis matrix"""
# pylint: disable = C0103

import pywt
import rwt
import numpy as np
import itertools
from argparse import ArgumentParser


def generate_basis(im_dim=64,db_num=1):
    """generate the basis"""
    x = np.zeros((im_dim, im_dim))
    coefs = pywt.wavedec2(x, 'db{}'.format(db_num))
    n_levels = len(coefs)
    basis = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1 #i-th unit vector
                        temp_basis = pywt.waverec2(coefs, 'db{}'.format(db_num))#apply wavelet decoder to e_i to get i-th column
                        basis.append(temp_basis)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    coefs[i][j][m] = 1
                    temp_basis = pywt.waverec2(coefs, 'db{}'.format(db_num))
                    basis.append(temp_basis)
                    coefs[i][j][m] = 0
    basis = np.array(basis)
    return basis

def best_singularvalues(W,m):#greedy
    n=W.shape[1]
    n_best=W.shape[0]
    print(n)
    Wtmp=W
    goodlist=np.array(range(n))
    for i in range(n-m):
        curr=0
        currj=0
        for j in goodlist:
            Wtmptmp=np.delete(Wtmp,j,axis=1)
            print(j)
            _,s,_=np.linalg.svd(Wtmptmp.T.dot(Wtmptmp))
            print(s[n_best-1])
            print(curr)
            print(currj)
            currj = j if s[n_best-1]>curr else currj
            curr = s[n_best-1] if s[n_best-1]>curr else curr  
            if s[n_best-1]<0.00001:
                goodlist = goodlist[goodlist!=j]
            print(len(goodlist))
        Wtmp = np.delete(Wtmp,currj,axis=1)
        goodlist=goodlist[goodlist!=currj]
        goodlist[goodlist>currj]=goodlist[goodlist>currj]-1
    _,s,_=np.linalg.svd(Wtmp.T.dot(Wtmp))
    return Wtmp,np.sqrt(s)

#PYTHON 3 FOR THIS PART!
def generate_basis_duplicate(mode,im_dim=64):
    """generate the basis"""
    #No downsampling of the scaling filter, full wavelet tree (wavelets from wavelets)
    x = np.zeros((im_dim, im_dim))
    basis_low = []
    basis_high = []
    scaling_filter, _ = rwt.daubcqf(im_dim, 'min')
    coefs = rwt.rdwt(x, scaling_filter)
    low_pass_coeff = coefs[0]
    high_pass_coeff = coefs[1]
    for i,j in itertools.product(range(low_pass_coeff.shape[0]),range(low_pass_coeff.shape[1])):
        low_pass_coeff[i][j] = 1
        filterij = rwt.irdwt(low_pass_coeff, high_pass_coeff, scaling_filter)[0]
        basis_low.append(np.reshape(filterij,(-1,1)))
        low_pass_coeff[i][j] = 0
    for i,j in itertools.product(range(high_pass_coeff.shape[0]),range(high_pass_coeff.shape[1])):
        high_pass_coeff[i][j] = 1
        filterij = rwt.irdwt(low_pass_coeff, high_pass_coeff, scaling_filter)[0]
        basis_high.append(np.reshape(filterij,(-1,1)))
        high_pass_coeff[i][j] = 0
    basis_low = np.squeeze(np.asarray(basis_low))   # Tensorflow expexcts the matrices transposed, so .T.T is required
    basis_high = np.squeeze(np.asarray(basis_high)) # like above
    if 'lowpass' in mode:
        basis = np.concatenate((basis_low,basis_high),axis=0)
    else:
        basis = basis_high
    return basis

def main(args):
    if 'redundant' in args.mode:
        basis = generate_basis_duplicate(args.mode,im_dim=args.pix)
    elif args.mode == 'nonredundant':
        basis = generate_basis(im_dim=args.pix)
    else:
        raise NotImplementedError
    np.save('./wavelet_basis-{}-{}.npy'.format(args.mode,args.pix), basis)

if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--mode', type=str, default='redundant', help='redundant or non-redundant')
    PARSER.add_argument('--pix', type=int, default=64, help='n, image has nxn pixels')
    args = PARSER.parse_args()
    print(args.mode)
    main(args)

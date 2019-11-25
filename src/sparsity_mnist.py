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
from argparse import ArgumentParser

#sparsity_representation('detail_sparsity','',True,refresh=True)
#sparsity_representation('detail_sparsity','TV',True,refresh=True)

def sparsity_representation(mode_data,mode,save,refresh=False,max_it='img_dim',norm_mode='standard',data_mode='mnist',eps=0):
    path = './estimated/{}/{}/'.format(data_mode,mode_data)
    img = load_data(data_mode)
    if norm_mode=='standard':
        norm='l2'
    else:
        norm=norm_mode
    addendum=data_mode+''+norm
    W='Id'
    if mode == 'TV':
        import utils_lasso
        dim_n=img.shape[1]
        gradX,gradY = utils_lasso.constr_grad(int(np.sqrt(dim_n)),mode='diff1_periodic')
        grad = np.concatenate((gradX,gradY))
        img = img.dot(grad.T)    
#        plt.figure()
#        plt.imshow(np.reshape(img[0],(28,-1)))
#        plt.imshow(np.reshape(img[0],(-1,28)))
#        plt.show()
#        plt.imshow(np.reshape(img[1],(28,-1)))
#        plt.imshow(np.reshape(img[1],(-1,28)))
#        plt.show()
#        plt.imshow(np.reshape(img[2],(28,-1)))
#        plt.imshow(np.reshape(img[2],(-1,28)))
#        plt.show()
        if norm_mode=='standard':
            norm='l12_mix'#former l1
        else:
            norm=norm_mode
        addendum = data_mode+'TV_'+norm
    if mode == 'wavelet':
        W = np.load('./framewavehaar.npy')
        Winv = np.linalg.inv(W)
        img = img.dot(Winv.T)
        addendum = data_mode+'Wav_'+norm
    if mode_data == 'mean_sparsity':
        mean_sparsity(img,save,eps=eps)
    elif mode_data == 'detail_sparsity':
        print(norm)
        sparsity_plot(img,save,refresh,addendum=addendum,max_it=max_it,norm=norm,path=path,W=W)
    
    
def load_data(data_mode):
    if data_mode == 'mnist':
        mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
        img = mnist.test.images #sampl_numb x img_dim
    elif data_mode == 'fashion-mnist':
        mnist = load_mnist('fashion-mnist')
        img = mnist.test.images
    else:
        raise NotImplementedError
    return img
    
def approx(W,x,k):#squared l2
    z=np.linalg.solve(W,x)
    return np.sum(np.sort((z**2))[:-k])


def check_bestkTermApprox(kterm, W, x, treat_as_if_ONB = True):#rows of W are phi_i
    if treat_as_if_ONB:
        W = W[:784,:784].T
    if W.shape[0] == W.shape[1]:
        if np.linalg.norm(W.dot(W.T)-np.eye(W.shape[0]))/(W.shape[0]**2) <=0.01: #check wether ONB
            Wx = W.dot(x)
            ind = np.argsort(Wx)[:kterm]
            xhat = W.T[:,ind].dot(Wx[ind])
            return xhat
        warnings.warn("W should be an ONB, but {} is not orthogonal or not normed!".format(W))
    warnings.warn("W should be an ONB, but {} is not quadratic (has shape {})!".format(W,W.shape))
    print('Starting brute force approach!')
    import itertools
    x_dict = {}
    val_dict = {}
    #for k in range(kterm):
    x_dict.update({i:W[:,list(i)].dot(np.linalg.solve(W[:,list(i)]),x) for i in itertools.combinations(range(W.shape[1]),kterm)})
    val_dict.update({i:np.linalg.norm(xhat-x) for i,xhat in x_dict.iteritems()})
    return x_dict[min(val_dict, key=val_dict.get)]

def load_wavelet(redundancy = 'nonredundant',pix=28):
    W = np.load('./wavelet_basis-{}-{}.npy'.format(redundancy,pix)).reshape(-1,784)
    W = np.asarray(W).T
    return W

def mutual_coherence(W):
    W = W/np.linalg.norm(W,axis=0)[:,np.newaxis].T
    coh = np.abs(W.T.dot(W))
    np.fill_diagonal(coh,0)
    return np.max(coh,axis=0)#np.max(coh)

#Returns average level of sparsity    
def mean_sparsity(img,save,eps=0):
    if eps >0:     
        img[np.abs(img)<=eps] = 0 
    spars_mean = np.mean(np.count_nonzero(img,axis=1))
    print(spars_mean)
    if save == True:
        path = './estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        np.save(path+'mean_sparsity.npy',spars_mean)
    return spars_mean


#calculation function
#Input:
#img: array with *all* data
#save: Whether to save the outcome
#refresh: Force new calculation
#maxit: maximal sparsity level. Default: img_dim
def sparsity_plot(img,save,refresh,addendum='',max_it='img_dim',norm='l2',path='./estimated/mnist/pure_sparsity/',W='Id'):
    img_dim = len(img[0])
    img_numb = len(img)
    plot_yes=True
    if plot_yes:
        from matplotlib import pyplot as plt
    #print(img_numb)
    if max_it == 'img_dim':
        max_it=int(img_dim)
    it_max = min(img_dim,max_it)
    
    flag, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_std,rep_err_pixel_euclid,rep_err_pixel_euclid_std,metr,interv,val,z_dim = chkpoint(path+addendum,it_max,refresh=refresh)
    if metr.shape[0]==0:
        metr = np.zeros((it_max,img_numb))
        interv = np.zeros((it_max))
        val = np.zeros((it_max))
    
    sortedArr=np.argsort(np.abs(img),axis=-1)
    indices = (np.ones((img_numb,img_dim))*np.asarray(range(img_numb))[:,None]).astype(int)
    if flag:
        for k in range(it_max):#could be massively improved
            ind_j = sortedArr[:,:img_dim-k]
            ind_i = indices[:,:img_dim-k]
            img_k = np.copy(img)
            img_k[ind_i,ind_j]=0 #one can store this
            print('average sparsity of z is {}'.format(np.sum(img_k!=0)/img_numb)) #sanity check
            if k==100:
                np.save('TESTZ.npy',img_k)
            if W=='Id':
                x_k=img_k
                if norm=='l2':#compute the pixel's mean for each data point
                    metric = np.sum((img_k-img)**2, axis = 1)
                elif norm=='l1':
                    metric = np.sum(np.abs(img_k-img), axis = 1)
                elif norm=='l12_mix':#wrong definition, not l2 norm of small blocks in l1, but l1 norm of small blocks in l2 norm
                    D = (img_k-img)**2
                    D = D[:,:int(img_dim/2)]+D[:,int(img_dim/2):]
                    D=np.sqrt(D)
                    metric = np.sum(D,axis=1)
            else:
                print('Synthesis')
                x_k=img_k.dot(W.T)
                if norm=='l2':#compute the pixel's mean for each data point
                    metric = np.sum((img_k.dot(W.T)-img.dot(W.T))**2, axis = 1)
                elif norm=='l1':
                    metric = np.sum(np.abs(img_k.dot(W.T)-img.dot(W.T)), axis = 1)
                elif norm=='l12_mix':#wrong definition, not l2 norm of small blocks in l1, but l1 norm of small blocks in l2 norm
                    D = (img_k.dot(W.T)-img.dot(W.T))**2
                    D = D[:,:int(img_dim/2)]+D[:,int(img_dim/2):]
                    D=np.sqrt(D)
                    metric = np.sum(D,axis=1)                
                #metric = np.sqrt(np.sum(D[:,:int(img_numb/2)],axis=1))+np.sqrt(np.sum(D[:,int(img_numb/2):],axis=1))
            #print metric.shape
            if plot_yes and k%5==0 and save:
                import utilsM
#                f = plt.figure()
#                plt.imshow(x_k[1,:].reshape(28,28),cmap='gray')
#                plt.savefig(path+ addendum +'bestkterm_{}.png'.format(k))
#                plt.close(f)
                mkdir_p(path)
                utilsM.save_images(np.reshape(x_k[:64,:],[-1,28,28]),[8,8],path+ addendum +'bestkterm_{}.png'.format(k))
            print('Iteration: {}'.format(k))            
            rep_err[k] = np.mean(metric)
            print('The mean representation error is: {}'.format(rep_err[k]))
            rep_err_std[k] = np.std(metric)#scaled in plot
            print('The stdev of representation error is: {}'.format(rep_err_std[k]))
            rep_err_pixel[k] = np.mean(metric/img_dim)#For gradient: division by gradient dimension
            print('The mean of pixwise representation error is: {}'.format(rep_err_pixel[k]))
            rep_err_pixel_std[k] = np.std(metric/img_dim)#/np.sqrt(img_numb)
            print('The stv of pixewise representation error is: {}'.format(rep_err_pixel_std[k]))
            rep_err_pixel_euclid[k] = np.mean(np.sqrt(metric)/img_dim)
            print('The mean pixewise squarerooted representation error is: {}'.format(rep_err_pixel_euclid[k]))
            rep_err_pixel_euclid_std[k] = np.std(np.sqrt(metric)/img_dim)#/np.sqrt(img_numb)
            print('The stdv representation squarerooted error is: {}'.format(rep_err_pixel_euclid_std[k]))    
            print(z_dim)
            interv[k],val[k] = points_contained(metric,thresh=0.5)
            metr[k] = metric
            print('When scaling the standard deviation by {}, {} percent of the data points lie inside the gained interval around the mean value'.format(interv[k],val[k]*100))
            if rep_err[k]==0:
                it_max = k
                break
       
    plot_error(path, rep_err, save, 'l2', 0,addendum=addendum)
    plt.figure()
    plot_errorbar(path, rep_err, rep_err_std, save, 'l2', 0)
    plt.show()
    plot_error(path, rep_err_pixel,save,'pixelwise_wo_sqrt',0,addendum=addendum)
    plot_error(path, rep_err_pixel_euclid,save,'pixelwise_with_sqrt',0,addendum=addendum)
    if save:        
        mkdir_p(path)
        np.save(path + addendum + 'rep_err.npy', rep_err)
        np.save(path + addendum + 'rep_err_std.npy', rep_err_std)
        np.save(path + addendum + 'rep_err_pixel.npy', rep_err_pixel)
        np.save(path + addendum + 'rep_err_pixel_std.npy', rep_err_pixel_std)
        np.save(path + addendum + 'rep_err_pixel_euclid.npy', rep_err_pixel_euclid)
        np.save(path + addendum + 'rep_err_pixel_euclid_std.npy', rep_err_pixel_euclid_std)
        np.save(path + addendum + 'metr.npy', metr)
        np.save(path + addendum + 'interv.npy', interv)
        np.save(path + addendum + 'val.npy', val)
        np.save(path + addendum + 'z_dim.npy', img_dim)
        print(path + addendum + 'rep_err')


#Call to either get stored values or to initialize the new ones
def chkpoint(path,it_max,refresh=False):    
    retrieval1 = ['rep_err','rep_err_std','rep_err_pixel','rep_err_pixel_std','rep_err_pixel_euclid','rep_err_pixel_euclid_std']
    retrieval2 = ['metr','interv','val','z_dim']
    retrieval = [name+'.npy' for name in retrieval1+retrieval2]
    test = np.asarray([os.path.isfile(path+name) for name in retrieval]).all()
    storagearr=[]
    if (not(refresh)):# and test):
        print(os.getcwd())
        storagearr.append(False)
        for name in retrieval:
            try:
                storagearr.append(np.load(path+name))
            except:
                print('could not load {}'.format(name))
#        rep_err = np.load(path + 'rep_err.npy')
#        rep_err_std = np.load(path + 'rep_err_std.npy')
#        rep_err_pixel = np.load(path + 'rep_err_pixel.npy')
#        rep_err_pixel_std = np.load(path + 'rep_err_pixel_std.npy')
#        rep_err_pixel_euclid = np.load(path + 'rep_err_pixel_euclid.npy')   
#        rep_err_pixel_euclid_std = np.load(path + 'rep_err_pixel_euclid_std.npy')   
#        metr = np.load(path + 'metr.npy')
#        interv = np.load(path + 'interv.npy')
#        val = np.load(path + 'val.npy')
#        return False, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_std, rep_err_pixel_euclid,rep_err_pixel_euclid_std,metr
    else:
        storagearr.append(True)
        for _ in range(len(retrieval1)):
            storagearr.append(np.zeros(it_max,))
#        rep_err = np.zeros(it_max,)
#        rep_err_std = np.zeros(it_max,)
#        rep_err_pixel = np.zeros(it_max,)
#        rep_err_pixel_std = np.zeros(it_max,)
#        rep_err_pixel_euclid = np.zeros(it_max,)
#        rep_err_pixel_euclid_std = np.zeros(it_max,)
        for _ in range(len(retrieval2)):
            storagearr.append(np.array([]))
#        metr = np.array([])
#        return True, rep_err, rep_err_std, rep_err_pixel, rep_err_pixel_std, rep_err_pixel_euclid,rep_err_pixel_euclid_std,metr
    return tuple(storagearr)

def gain_mean_std(metr,max_it='img_dim',mode='eucl_pix',img_dim=784):
    metr = np.asarray(metr).astype(np.float)
    #img_dim = metr.shape[0]
   # metr_numb = metr.shape[1]
    #print(img_numb)
    if max_it == 'img_dim':
        max_it=img_dim
    it_max = min(img_dim,max_it)
    interv = np.zeros((it_max,))
    std_val = np.zeros((it_max,))
    val = np.zeros((it_max,))
    mean_val= np.zeros((it_max,))
    if 'eucl' in mode:
        metr = np.sqrt(metr)
    if 'pix' in mode:
        metr=metr/img_dim
    for k in range(it_max):
        mean_val[k] = np.mean(metr[k])
        interv[k],val[k] = points_contained(metr[k],thresh=0.5)
        std_val[k] = np.std(metr[k])*interv[k]       
    return (mean_val,std_val)

def plot_error(path, data,save,name,lim,addendum=''):
    plt.figure()
    plt.plot(range(len(data)),data)
    if lim!='Auto':
        plt.ylim(bottom=lim)
    if save == True:
        #path = '../estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        plt.savefig(path+ addendum +'error_plot_sparsity_{}.pdf'.format(name))
        np.save(path+ addendum +'error_plot_sparsity_{}.npy'.format(name),data)
    plt.show()

def plot_errorbar(path, data,stdv, save,name,lim,ax='',color=None,ecolor=None,indices=[250, 260,270,280,290,300, 400],addendum=''):
#    print(np.array(data))
    #indices = [5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500, 783]#range(len(data))
    data = np.array(data)[np.array(indices)]
    stdv = np.array(stdv)[np.array(indices)]    
    if ax=='':
#        plt.errorbar(indices, data, yerr=1.96*stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
        plt.errorbar(indices, data, yerr=stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
    else:
#        ax.errorbar(indices, data, yerr=1.96*stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
        ax.errorbar(indices, data, yerr=stdv, marker='o', markersize=5, capsize=5,color=color,ecolor=ecolor)
    if lim!='Auto':
        plt.ylim(bottom=lim)
    if save == True:
        #path = '../estimated/mnist/pure_sparsity/'
        mkdir_p(path)
        plt.savefig(path+ addendum +'error_plot_sparsity_{}.pdf'.format(name))
        np.save(path+ addendum +'error_plot_sparsity_{}.npy'.format(name),data)
    
        
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def points_contained(cut,thresh=0.5):
    #print cut
    if len(cut)==0:
        return (np.NaN,np.NaN)
    cut = np.asarray(cut)
    m = np.mean(cut)
    n = float(len(cut))
    s=np.std(cut)
    flag = True
    i=0.
    j=2. #if indeed random variable, should be 75% concentrated at least
#    p=float(j/2)
    while flag:
        p=i+(j-i)/2
#        print(p)        
        crit = len(cut[np.abs(cut-m)<=p*s])/n
#        print(crit)
        if crit>=thresh:
            j=p
        else:
            i=p
        if np.abs(i-j) <= 0.01:
            flag=False
            if crit<thresh:
                p=j
                crit = len(cut[np.abs(cut-m)<=p*s])/n
    return (p,crit)
#-----------------------------------from mnist_input---------------------------------------------------

def load_mnist(dataset_name):
    import os,gzip
    import numpy as np
    class data_container(object):
        class container_field(object):
            def next_batch(self,batch):
                ret_valx = self.images[self.pos:self.pos+batch]
                ret_valy = self.labels[self.pos:self.pos+batch]
                self.pos = self.pos + batch
                return (ret_valx,ret_valy)
            def __init__(self):
                self.pos=0
        def __init__(self,training_data,training_label,test_data,test_label):
            test = data_container.container_field()
            test.images = test_data
            test.labels = test_label
            training = data_container.container_field()
            training.images = training_data
            training.labels = training_label
            self.test = test
            self.train = training
            
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = np.float32(np.asarray(data.reshape((60000, -1)))/255)

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000)).astype(int)

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = np.float32(np.asarray(data.reshape((10000, -1)))/255)

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000)).astype(int)

    y_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec[i, teY[i]] = 1.0
    teY = y_vec
        
    y_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec[i, trY[i]] = 1.0    
    trY = y_vec
    return data_container(trX,trY,teX,teY)
#-----------------------------------------------------------------------
#sparsity_representation('mean_sparsity','',True,refresh=True)
#sparsity_representation('mean_sparsity','TV',True,refresh=True)
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784,norm_mode='l1')
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784)
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784,norm_mode='l2')
#sparsity_representation('detail_sparsity','',True,refresh=True,max_it=784)
#sparsity_representation('mean_sparsity','',True,refresh=True,data_mode='fashion-mnist')
#sparsity_representation('mean_sparsity','TV',True,refresh=True,data_mode='mnist',eps=0.001)
#sparsity_representation('mean_sparsity','TV',True,refresh=True,data_mode='fashion-mnist')
#sparsity_representation('mean_sparsity','TV',True,refresh=True,data_mode='fashion-mnist',eps=0.01)
#sparsity_representation('mean_sparsity','wavelet',True,refresh=True,data_mode='mnist',eps=0.001)
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784,norm_mode='l1',data_mode='fashion-mnist')
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784,data_mode='fashion-mnist')
#sparsity_representation('detail_sparsity','TV',True,refresh=True,max_it=784,norm_mode='l2',data_mode='fashion-mnist')
#sparsity_representation('detail_sparsity','',True,refresh=True,max_it=784,data_mode='fashion-mnist')
#
#sparsity_representation('detail_sparsity','wavelet',True,refresh=True,data_mode='mnist')
#sparsity_representation('detail_sparsity','wavelet',True,refresh=True,data_mode='fashion-mnist')
if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--mode_data', type=str, default='detail_sparsity',help='compute only the mean_sparsity (average (analysis/synthesis) sparsity) or detail_sparsity (best k-approximation (heuristic))')
    PARSER.add_argument('--mode', type=str, default='pure', help='Set to TV or wavelet, if these modes are desired')
    PARSER.add_argument('--save', action='store_true', help='whether to save the stats')
    PARSER.add_argument('--refresh', action='store_true', help='whether to refresh the data or use pre-computed ones, if existing')
    PARSER.add_argument('--max_it', type=str, default='img_dim', help='Upto which k should the best k term approx be computed')
    PARSER.add_argument('--norm_mode', type=str, default='standard', help='Which norm to use for the metric: l2,l1 or l12_mix, standard (choosing it dependent on mode automatically)')
    PARSER.add_argument('--data_mode', type=str, default='mnist', help='Which data to use (mnist or fashion-mnist)')
    PARSER.add_argument('--eps', type=float, default=0, help='Tolerance for mean_sparsity (what counts as "zero")')
    PARSER=PARSER.parse_args()
    sparsity_representation(PARSER.mode_data,PARSER.mode,PARSER.save,refresh=PARSER.refresh,max_it=PARSER.max_it,norm_mode=PARSER.norm_mode,data_mode=PARSER.data_mode,eps=PARSER.eps)
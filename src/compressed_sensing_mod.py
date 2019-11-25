#!/usr/bin/env python2
"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
print(os.system('pwd'))
from argparse import ArgumentParser
import numpy as np
import utils
import matplotlib.pyplot as plt
import copy
import mnist_model_def
import utilsM
import time
import datetime
import warnings


def main(hparams):
#    if not hparams.use_gpu:
#        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    #hparams.stdv = 10 #adjust to HPARAM in model_def.py
    #hparams.mean = 0 #adjust to HPARAM in model_def.py
    utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    hparams.bol = False
 #   hparams.dict_flag = False
    # get inputs
    if hparams.input_type == 'dict-input':# or hparams.dict_flag:
        hparams_load_key = copy.copy(hparams)
        hparams_load_key.input_type = 'full-input'
        hparams_load_key.measurement_type = 'project'
        hparams_load_key.zprior_weight = 0.0
        hparams.key_field = np.load(utils.get_checkpoint_dir(hparams_load_key, hparams.model_types[0])+'candidates.npy').item()
        print(hparams.measurement_type)
    xs_dict, label_dict = model_input(hparams)    

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    sh = utils.SaveHandler()
    sh.load_or_init_all(hparams.save_images,hparams.model_types,sh.get_pkl_filepaths(hparams,use_all=True))
    if label_dict is None:
        print('No labels exist.')
        del sh.class_loss
#    measurement_losses, l2_losses, emd_losses, x_orig, x_rec, noise_batch = utils.load_checkpoints(hparams)
    
    if hparams.input_type == 'gen-span':
        np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'z.npy',hparams.z_from_gen)
        np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'images.npy',hparams.images_mat)
    
    

    x_hats_dict = {model_type : {} for model_type in hparams.model_types}
    x_batch_dict = {}
    x_batch=[]
    x_hat_batch=[]
#    l2_losses2=np.zeros((len(xs_dict),1))
#    distances_arr=[]
    image_distance =np.zeros((len(xs_dict),1))
    hparams.x = [] # TO REMOVE
    for key, x in xs_dict.iteritems(): #//each batch once (x_batch_dict emptied at end)
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        x_batch_dict[key] = x       
        hparams.x.append(x)#To REMOVE
        if len(x_batch_dict) < hparams.batch_size:
            continue
        
        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.iteritems()]
        x_batch = np.concatenate(x_batch_list)
#        x_batch, known_distortion, distances = get_random_distortion(x_batch)
#        distances_arr[(key-1)*hparams.batch_size:key*hparams.batch_size] = distances
#        xs_dict[(key-1)*hparams.batch_size:key*hparams.batch_size] =x_batch
        
        # Construct noise and measurements
        recovered, optim = utils.load_if_optimized(hparams)
        if recovered and np.linalg.norm(optim.x_orig-x_batch) < 1e-10:
            hparams.optim = optim
            hparams.recovered = True
        else:
            hparams.recovered=False
            optim.x_orig = x_batch
            
            hparams.optim = optim
            
        A, noise_batch, y_batch, c_val = utils.load_meas(hparams,sh,x_batch,xs_dict)
        hparams.optim.noise_batch = noise_batch
        if c_val:
            continue
        
        if hparams.measurement_type == 'sample_distribution':
            plot_distribution(hparams,x_batch)
            
#            for i in range(z.shape[1]):#range(1):
#                plt.hist(z[i,:], facecolor='blue', alpha=0.5)
#                directory_distr = 
#                pl.savefig("abc.png")            
        elif hparams.measurement_type == 'autoencoder':
            plot_reconstruction(hparams,x_batch) 
        else:
            # Construct estimates using each estimator
            for model_type in hparams.model_types:
                estimator = estimators[model_type]
                start = time.time()

                tmp = estimator(A, y_batch, hparams)
                if isinstance(tmp,tuple):
                    x_hat_batch = tmp[0]
                    sh.z_rec = tmp[1]                    
                else:
                    x_hat_batch = tmp
                    del sh.z_rec
                end = time.time()
                duration = end-start
                print('The calculation needed {} time'.format(datetime.timedelta(seconds=duration)))
                np.save(utils.get_checkpoint_dir(hparams, model_type)+'elapsed_time',duration)
#                DEBUGGING = []
                for i, key in enumerate(x_batch_dict.keys()):
    #                x = xs_dict[key]+known_distortion[i]
                    x = xs_dict[key]
                    y = y_batch[i]
                    x_hat = x_hat_batch[i]
#                    plt.figure()
#                    plt.imshow(np.reshape(x_hat, [64, 64, 3])*255)#, interpolation="nearest", cmap=plt.cm.gray)
#                    plt.show()
    
                    # Save the estimate
                    x_hats_dict[model_type][key] = x_hat
    
                    # Compute and store measurement and l2 loss
                    sh.measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y)
#                    DEBUGGING.append(np.sum((x_hat.dot(A)-y)**2)/A.shape[1])
                    sh.l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                    if hparams.class_bol and label_dict is not None:
                        try:
                            sh.class_losses[model_type][key] = utils.get_classifier_loss(hparams,x_hat,label_dict[key])
                        except:
                            sh.class_losses[model_type][key] = NaN
                            warnings.warn('Class loss unsuccessfull, most likely due to corrupted memory. Simply retry.')
                    if hparams.emd_bol:
                        try:
                            _,sh.emd_losses[model_type][key] = utils.get_emd_loss(x_hat, x)
                            if 'nonneg' not in hparams.tv_or_lasso_mode and 'pca'  in model_type:
                                warnings.warn('EMD requires nonnegative images, for safety insert nonneg into tv_or_lasso_mode')
                        except ValueError:
                            warnings.warn('EMD calculation unsuccesfull (most likely due to negative images)')
                            pass
    #                    if l2_losses[model_type][key]-measurement_losses[model_type][key]!=0:
    #                        print('NO')
    #                        print(y)
    #                        print(x)
    #                        print(np.mean((x-y)**2))
                    image_distance[i] = np.linalg.norm(x_hat-x)
    #                l2_losses2[key] = np.mean((x_hat-x)**2)
    #                print('holla')
    #                print(l2_losses2[key])
    #                print(np.linalg.norm(x_hat-x)**2/len(xs_dict[0]))
    #                print(np.linalg.norm(x_hat-x)/len(xs_dict[0]))
    #                print(np.linalg.norm(x_hat-x))
            print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))
            sh.x_orig = x_batch
            sh.x_rec = x_hat_batch
            sh.noise = noise_batch
    
            #ACTIVATE ON DEMAND
            #plot_bad_reconstruction(measurement_losses,x_batch)
            # Checkpointing
            if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):           
                utils.checkpoint(x_hats_dict, save_image, sh, hparams)
                x_hats_dict = {model_type : {} for model_type in hparams.model_types}
                print('\nProcessed and saved first ', key+1, 'images\n')    
            x_batch_dict = {}
                   

    if 'wavelet' in hparams.model_types[0]:
        print np.abs(sh.x_rec)
        print('The average sparsity is {}'.format(np.sum(np.abs(sh.x_rec)>=0.0001)/float(hparams.batch_size)))

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(x_hats_dict, save_image, sh, hparams)
        print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))
        if hparams.dataset in ['mnist', 'fashion-mnist']:
            if np.array(x_batch).size:
                utilsM.save_images(np.reshape(x_batch, [-1, 28, 28]),
                                          [8, 8],utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'original.png')
            if np.array(x_hat_batch).size:
                utilsM.save_images(np.reshape(x_hat_batch, [-1, 28, 28]),
                                          [8, 8],utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'reconstruction.png')

        for model_type in hparams.model_types:
#            print(model_type)
            mean_m_loss = np.mean(sh.measurement_losses[model_type].values())
            mean_l2_loss = np.mean(sh.l2_losses[model_type].values()) #\|XHUT-X\|**2/784/64
            if hparams.emd_bol:
                mean_emd_loss = np.mean(sh.emd_losses[model_type].values())
            if label_dict is not None:
                mean_class_loss = np.mean(sh.class_losses[model_type].values())
                print('mean class loss = {0}'.format(mean_class_loss))
#            print(image_distance)
            mean_norm_loss = np.mean(image_distance)#sum_i(\|xhut_i-x_i\|)/64
#            mean_rep_error = np.mean(distances_arr)
#            mean_opt_meas_error_pixel = np.mean(np.array(l2_losses[model_type].values())-np.array(distances_arr)/xs_dict[0].shape)
#            mean_opt_meas_error = np.mean(image_distance-distances_arr)
            print('mean measurement loss = {0}'.format(mean_m_loss))
#            print np.sum(np.asarray(DEBUGGING))/64
            print('mean l2 loss = {0}'.format(mean_l2_loss))
            if hparams.emd_bol:
                print('mean emd loss = {0}'.format(mean_emd_loss))            
            print('mean distance = {0}'.format(mean_norm_loss))
            print('mean distance pixelwise = {0}'.format(mean_norm_loss/len(xs_dict[xs_dict.keys()[0]])))
#            print('mean representation error = {0}'.format(mean_rep_error))
#            print('mean optimization plus measurement error = {0}'.format(mean_opt_meas_error))
#            print('mean optimization plus measurement error per pixel = {0}'.format(mean_opt_meas_error_pixel))

    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
        print('Consider rerunning lazily with a smaller batch size.')

#def get_Q(hparams):
def get_random_distortion(batch):
    size_batch = batch.shape
    #distortion = -np.abs(np.random.uniform(np.zeros(size_batch),batch,size_batch))
    distortion_bound = batch
    distortion_bound[distortion_bound<=0.5]=0
    distortion_bound[distortion_bound>0.5]=1
    distortion = -np.abs(np.random.uniform(np.zeros(size_batch),distortion_bound,size_batch))
    print(distortion)
    norm_distortion = np.array([np.linalg.norm(x) for x in distortion]) #numerical feast.. not
#    print(norm_distortion.shape)
#    print(distortion.shape)
    return (distortion_bound,distortion,norm_distortion)

#def normalize(x):
#    if np.linalg.norm(x)>0:
#        return x/np.linalg.norm(x)
#    else:
#        return x
    
def plot_bad_reconstruction(measurement_losses,x_batch):
    for l in measurement_losses.keys():
        ran = np.ceil(len(measurement_losses[l])*0.1).astype('int32')
        rec_percentiles = np.ones((10,ran))
        convert = [v for (k,v) in measurement_losses[l].items()]
        ind = np.argsort(convert)
        for i in range(len(ind)):
            i1=np.floor(i/ran).astype('int32')
            i2 = np.floor(i % ran).astype('int32')
            result = np.linalg.norm(x_batch[measurement_losses[l].keys()[ind[i]]])
            rec_percentiles[i1,i2] = result
        plt.bar(range(10),np.mean(rec_percentiles,1))
        plt.show()
        print(measurement_losses)



def plot_candidates(data):
    for i  in range(len(data)):
        plt.figure()
        plt.imshow(data[i])
        plt.show()
                
def plot_distribution(hparams,x_batch):
    #hparams.n_z=20
    z = applicate_encoder(hparams,x_batch)
    print(z.shape)
    for j in range(hparams.n_z):
        plt.subplot(2,10,j+1)
        _,bins,_=plt.hist([z[i,j] for i in range(z.shape[0])],alpha=0.5,density=True)
        width = np.abs(bins[0]-bins[1])
        exp_bins = [i*width-3 for i in range(np.ceil(6/width).astype('int32'))]
        plt.plot(exp_bins,[np.exp(-1/2*n**2)/(np.sqrt(2*np.pi)) for n in exp_bins],alpha=0.5)
        plt.xlim([-3,3])
    plt.show()

def plot_reconstruction(hparams,x_batch):
#    per_line = 8
    z = applicate_encoder(hparams,x_batch)
    x_r = applicate_decoder(hparams,z)
#    final_img = np.eye(28*per_line)
#    final_imgI = np.eye(28*per_line)
#    for i in range(x_batch.shape[0]):
##        save_image(np.reshape(x_r[i],(28,28)),utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'rec_reconstr_{}'.format(i))
##        save_image(np.reshape(x_batch[i],(28,28)),utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'rec_orig{}'.format(i))
#        sl_low = int(np.trunc(i/per_line)*28)
#        sl_low_fast = int((np.trunc(i % per_line))*28)
#        sl_up = int(np.trunc(i/per_line+1)*28)
#        sl_up_fast = int(np.trunc(i % per_line+1)*28)
#        final_img[sl_low:sl_up,sl_low_fast:sl_up_fast]  = np.reshape(x_batch[i],(28,28))
#        final_imgI[sl_low:sl_up,sl_low_fast:sl_up_fast] = np.reshape(x_r[i],(28,28))
#    save_image(final_img,utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'rec_orig{}'.format('all'))
#    save_image(final_imgI,utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'rec_reconstr{}'.format('all'))
    utilsM.save_images(np.reshape(x_batch, [-1, 28, 28]),
                                  [8, 8],'./orig_all_{}.png'.format(hparams.n_z))
    utilsM.save_images(np.reshape(x_r, [-1, 28, 28]),
                                  [8, 8],'./rec_all_{}.png'.format(hparams.n_z))
  
        
if __name__ == '__main__':
#    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='random-test', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebAtest/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=10, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=64, help='How many examples are processed together')
    PARSER.add_argument('--input-seed', type=str, default='no_seed', help='For random-test input mode fixes a seed')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--project_flag', action='store_true', help='Whether best of selection of the input should be done')
    PARSER.add_argument('--noise-std', type=float, default=0.0, help='std dev of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--inpaint-size', type=int, default=1, help='size of block to inpaint')
    PARSER.add_argument('--superres-factor', type=int, default=2, help='how downsampled is the image')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+',default=None, help='model(s) used for estimation') #nargs='+',
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=0.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior-weight', type=float, default=0.0, help='weight on z prior')
    PARSER.add_argument('--stdv', type=float, default=1.0, help='standard deviation of z')
    PARSER.add_argument('--mean', type=float, default=0.0, help='mean of z')
    PARSER.add_argument('--n-z', type=int, default=-1, help='hidden dimension of z')
    PARSER.add_argument('--tf_init', type=int, default=1, help='use tf or np')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')
    PARSER.add_argument('--grid', type=str, default="NoGrid", help='specifier for mnist vae flexible')
    PARSER.add_argument('--tol', type=int, default=5, help='precision in binary search for mnist vae flexible')
    PARSER.add_argument('--init-mode', type=str, default='random', help='initialization mode for estimator')
    PARSER.add_argument('--fair-counter', type=str, default='unequal', help='If and how many times the fixed version is reiterated to make up for the additional optimization')
    PARSER.add_argument('--fair-counter-end', type=int, default=1, help='If and how many times the final iteration is reiterated to improve the optimization')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='momentum', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')
    PARSER.add_argument('--eps', type=float, default=0.0002, help='eps for measurement for flex vae (weighted with norm of A)')
    PARSER.add_argument('--flex-chosen', type=str, default='flexible', help='fixed dimension of the VAE flex (good for projection)')
    PARSER.add_argument('--tolerance-checking', type=str, default='non-squared', help='Tolerance checking w.r.t. euclidian norm or squared euclidian norm')
    PARSER.add_argument('--strict-checking', type=str, default='strict', help='When using alternating checking, use only the grid points')
    PARSER.add_argument('--repetition-bol', type=str, default = 'False', help='Whether to repeat generator training as many times as grid points')


    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')
    PARSER.add_argument('--tv-or-lasso-mode', type=str, default='nada', help='cvxopt-constr, cvxopt-reg, fista,cvxpy-constr,cvxpy-reg')
    PARSER.add_argument('--omp-k', type=int, default=300, help='Orthogonal Matching Pursuit sparsity parameter')
    PARSER.add_argument('--kterm', type=int, default=-1, help='For representation system to make incomplete')
    PARSER.add_argument('--wavelet-type', type=str, default='', help='Which wavelet type to use')

    # k-sparse-wavelet specific hparams
    PARSER.add_argument('--sparsity', type=int, default=1, help='number of non zero entries allowed in k-sparse-wavelet')

    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print(statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')
    PARSER.add_argument('--matlab', type=str, default='', help='Wavelet case: Should use python generated or matlab generated wavelet systems')

    PARSER.add_argument('--use-gpu', type=str, default='False', help='Whether to use GPUs if possible')
    PARSER.add_argument('--reproducible', type=str, default='None', help='Whether the measurement matrix A is drawn with fixed seed')

    PARSER.add_argument('--emd-bol', type=str, default = 'True', help='emd loss logged')
    PARSER.add_argument('--class-bol', type=str, default = 'True', help='emd loss logged')
    

    HPARAMS = PARSER.parse_args()
    if HPARAMS.matlab=='nada':
        HPARAMS.matlab=''
    if 'wavelet' not in HPARAMS.model_types[0]:
        HPARAMS.wavelet_type=''    
    if HPARAMS.emd_bol == 'True':
       HPARAMS.emd_bol = True
    else:
       HPARAMS.emd_bol = False   
    if HPARAMS.class_bol == 'True':
       HPARAMS.class_bol = True
    else:
       HPARAMS.class_bol = False   
       
    if HPARAMS.flex_chosen == 'flexible' and HPARAMS.fair_counter != 'unequal':
        warnings.warn("----------------------------------------------------------------------------------------------------------------------------The fair_counter is thought to be a support for the fixed version. Do not use it for the flexible ones (except you really know what you're doing)!-------------------------------------------------------------------------------------------------")
    
#    HPARAMS.model_types = [HPARAMS.model_types] #downwards compatibility
    if HPARAMS.grid !='NoGrid':
        print('reset n-z because grid is not empty: {}'.format(HPARAMS.grid))
        HPARAMS.n_z = -1
        HPARAMS.grid = map(int,HPARAMS.grid.split())
#    if HPARAMS.n_z != -1:
#        print('Changed grid to n_z: {}.'.format(HPARAMS.n_z))
#        HPARAMS.grid = HPARAMS.n_z
    if HPARAMS.use_gpu == 'False':
        HPARAMS.use_gpu = False
    else:
        HPARAMS.use_gpu = True
    
    
    if HPARAMS.reproducible == 'None':
        HPARAMS.reproducible = ''
    if HPARAMS.kterm != -1:
       HPARAMS.tv_or_lasso_mode = 'kterm{}-'.format(HPARAMS.kterm) + HPARAMS.tv_or_lasso_mode
    dataname = HPARAMS.dataset
    print dataname
    if dataname in ['mnist','fashion-mnist']:        
        HPARAMS.image_shape = (28, 28)
        from mnist_input import model_input        
        from mnist_utils import view_image, save_image
#        print np.asarray([mode in HPARAMS.model_types[0] for mode in utils.get_mode_list()]).any()
#        print HPARAMS.model_types[0]
#        print utils.get_mode_list()
        if HPARAMS.model_types[0] in ['vae-flex']:
#            from mnist_vae_flex.src.model_def import applicate_encoder
#            from mnist_vae_flex.src.model_def import applicate_decoder
            if HPARAMS.grid!='':
                #HPARAMS.pretrained_model_dir = './mnist_vae_flex/models/mnist-vae-flex-{}/'.format(HPARAMS.grid[-1])
                if type(HPARAMS.grid)==list:                    
                    HPARAMS.pretrained_model_dir = './mnist_vae_flex/models/{}-vae-flex-{}/'.format(dataname,utils.process_grid(HPARAMS.grid))
                else:
                    HPARAMS.pretrained_model_dir = './mnist_vae_flex/models/{}-vae-flex-{}/'.format(dataname,HPARAMS.grid)
        elif HPARAMS.model_types[0] in ['vae-flex-alt']:
            print HPARAMS.pretrained_model_dir
            #HPARAMS.pretrained_model_dir = './mnist_vae_flex'+HPARAMS.pretrained_model_dir.split('./{}-vae-flex'.format(dataname))[1]
        elif np.asarray([mode in HPARAMS.model_types[0] for mode in utils.get_mode_list()]).any():
            #HPARAMS.pretrained_model_dir = '_'.join(HPARAMS.pretrained_model_dir.split('-'))
            pass
        else:
            from mnist_model_def import applicate_encoder
            from mnist_model_def import applicate_decoder
            if HPARAMS.n_z!=-1:               
                HPARAMS.pretrained_model_dir = './mnist_vae/models/{}-vae{}/'.format(dataname,HPARAMS.n_z)
            else:
                #pass
                HPARAMS.pretrained_model_dir = './mnist_vae/models/{}-vae{}/'.format(dataname,HPARAMS.grid[-1])

            
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError
    print(HPARAMS.stdv)
    print(HPARAMS.mean)
    main(HPARAMS)

"""Some common utils"""
# pylint: disable = C0301, C0103, C0111

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
try:
    import scipy.misc
    from lightning.regression import FistaRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso
    from l1regls import l1regls
    from cvxopt import matrix
    import ot
    import ot_unbalanced
    import mnist_estimators
    import mnist_estimators_flexI
    import celebA_estimators
except:
    pass    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import re

import itertools
import warnings



#----------------------------------------------------------
#def save_images(images, size, image_path):
#    return imsave(inverse_transform(images), size, image_path)
#
#
#def imsave(images, size, path):
#    return scipy.misc.imsave(path, merge(images, size))
#
#
#def inverse_transform(images):
#    return (images+1.)/2
#
#
#def merge(images, size):
#    h, w = images.shape[1], images.shape[2]
#    img = np.zeros((h * size[0], w * size[1]))
#    for idx, image in enumerate(images):
#        i = idx % size[1]
#        j = idx // size[1]
#        img[j*h:j*h+h, i*w:i*w+w] = image
#    return img
#
#
#def print_hparams(hparams):
#    print('')
#    for temp in dir(hparams):
#        if temp[:1] != '_':
#            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
#    print('')
#
#
#def set_up_dir(directory, clean=False):
#    if os.path.exists(directory):
#        if clean:
#            shutil.rmtree(directory)
#    else:
#        os.makedirs(directory)
#
#
#def get_ckpt_path(ckpt_dir):
#    ckpt_dir = os.path.abspath(ckpt_dir)
#    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
#    if ckpt and ckpt.model_checkpoint_path:
#        ckpt_path = os.path.join(ckpt_dir,
#                                 ckpt.model_checkpoint_path)
#    else:
#        ckpt_path = None
#    return ckpt_path
#
#
#def try_restore(hparams, sess, model_saver):
#    """Attempt to restore variables from checkpoint"""
#    ckpt_path = get_ckpt_path(hparams.ckpt_dir)
#    if ckpt_path:  # if a previous ckpt exists
#        model_saver.restore(sess, ckpt_path)
#        start_epoch = int(ckpt_path.split('/')[-1].split('-')[-1])
#        print('Succesfully loaded model from {0} at counter = {1}'.format(
#            ckpt_path, start_epoch))
#    else:
#        print('No checkpoint found')
#        start_epoch = -1
#    return start_epoch



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
#----------------------------------------------------------------------------------------------------------------

class BestKeeper(object):#Only l2 measurement loss counts
    """Class to keep the best stuff"""
    def __init__(self, hparams,logg_z=False):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))
        self.z_batch_val_best = np.zeros((hparams.batch_size, hparams.grid[-1]))
        self.logg_z=logg_z

    def report(self, x_hat_batch_val, losses_val,z_val='no_save'):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]
                if self.logg_z:
                    self.z_batch_val_best[i,:]=0
                    self.z_batch_val_best[i,:z_val.shape[1]] = z_val[i,:]

    def get_best(self):
        if self.logg_z:
            return self.x_hat_batch_val_best, self.z_batch_val_best
        else:
            return self.x_hat_batch_val_best

def load_if_optimized(hparams):
    #check whether already optimized
    if not os.path.exists(get_checkpoint_dir(hparams, hparams.model_types[0])+'tmp/'):
        os.mkdir(get_checkpoint_dir(hparams, hparams.model_types[0])+'tmp/')
    if os.path.exists(get_checkpoint_dir(hparams, hparams.model_types[0])+'tmp/optim.pkl'):
        recovered=True
        optim = np.load(get_checkpoint_dir(hparams, hparams.model_types[0])+'tmp/optim.pkl',allow_pickle=True)
    else:        
        recovered=False
        import utils_vaeflex
        optim = utils_vaeflex.Optim()
   
    #load x, A, z, iteration, loss, global_step
    
    #update BestKeeper, num_it and j
    
    return (recovered,optim)

def load_meas(hparams,sh,x_batch,xs_dict,cont_val = False):
    A = get_A(hparams)
    np.save(get_checkpoint_dir(hparams, hparams.model_types[0])+'A.npy',A)        
    noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
    if hparams.measurement_type == 'project' and hparams.project_flag==False:
        y_batch = x_batch + noise_batch
    elif hparams.measurement_type == 'project' and hparams.project_flag==True:
        (candidates,l2_loss) = get_close_candidates(hparams,xs_dict)
        print('Let us validate')
        import mnist_model_def
        mnist_model_def.validate(hparams.z_from_gen,hparams.images_mat,hparams)
        #plot_candidates(candidates)
        print(l2_loss)
        np.save(get_checkpoint_dir(hparams, hparams.model_types[0])+'candidates.npy',candidates)
        np.save(get_checkpoint_dir(hparams, hparams.model_types[0])+'candidates_l2_loss.npy',l2_loss)
        sh.l2_losses[hparams.model_types[0]]=l2_loss
        cont_val = True
    else:
#            y_batch = np.matmul(x_batch+known_distortion, A) + noise_batch
        y_batch = np.matmul(x_batch, A) + noise_batch
        
    if hparams.recovered:
        optim = hparams.optim
        A = optim.A
        noise_batch = optim.noise_batch
        y_batch = optim.y_batch
    return (A,noise_batch,y_batch,cont_val)

def get_close_candidates(hparams,dictionary, repeats=2, tol=0.001, minNumb=64):   
    hparams.bol=True
    tmp_batch_size = []
    tmp_batch_size.append(hparams.batch_size)
    tmp_num = hparams.max_update_iter
    tmp_res = hparams.num_random_restarts
    for it in range(repeats):
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in dictionary.iteritems()]
        x_batch = np.concatenate(x_batch_list)
        print(it)
        if it==repeats-1:
            tmp_num = hparams.max_update_iter
            tmp_res = hparams.num_random_restarts
            hparams.num_random_restarts = max(hparams.num_random_restarts,2)
            hparams.max_update_iter = max(2000,hparams.max_update_iter)
        if x_batch.shape[0]!=hparams.batch_size:
            tmp_batch_size.append(hparams.batch_size)
            hparams.batch_size = x_batch.shape[0]
            print('Smaller INPUT NUMBER: {} instead of {}'.format(x_batch.shape[0],tmp_batch_size[-1]))#Or change hparams on the fly
        estimators = get_estimators(hparams)
        estimator = estimators[hparams.model_types[0]]
        x_batch_approx = estimator(None,x_batch, hparams)
        losses = [get_l2_loss(x_batch_approx[i], x_batch[i]) for i in range(len(x_batch))]
        tol_losses = [idx for idx in range(len(losses)) if losses[idx] <= tol]
        if len(tol_losses) > len(dictionary)/10:
            keys = dictionary.keys()
            dictionary = dict((keys[i],dictionary[keys[i]]) for i in tol_losses)
            l2_losses = dict((keys[i],losses[i]) for i in tol_losses)
            break
#            return dict((key,dictionary[key]) for key in dictionary.keys()[tol_losses])
        else:
            ind = np.argsort(losses)[:max(np.ceil(len(losses)/10).astype('int32'),minNumb)]
            keys = dictionary.keys()
            dictionary = dict((keys[i],dictionary[keys[i]]) for i in ind)
            l2_losses = dict((keys[i],losses[i]) for i in ind)
    hparams.max_update_iter = tmp_num
    hparams.num_random_restarts = tmp_res
    hparams.batch_size = tmp_batch_size[0]
    return (dictionary,l2_losses)

def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)
#sum((i1_j-i2_j)**2)/784 =\|i1-i2\|**2/784
#
 
def scale(im):
    im = im - np.min(np.min(im),0)
#    if np.max(im)==0:
#        scale = 1
#    else:
#        scale = np.max(im)
#    im = im/scale    
    return im
    
def get_emd_loss(image1,image2,enforce_distr=True):
    if enforce_distr:
        image1 = scale(image1)
        image2 = scale(image2)
    choice = 'pot_sinkhorn_unbalanced'    
    if image1.ndim == 1:
        d = int(np.sqrt(len(image1)))
        image1 = np.reshape(image1,(d,d))
        image2 = np.reshape(image2,(d,d))
    numerical_weight = 1# np.product(image1.shape)
    if True:#choice != 'pot_sinkhorn_unbalanced':
        w1 = np.sum(image1)
        w2 = np.sum(image2)
        eps = 1e-10
        if (image1<0).any() or (image2<0).any():
            np.save('error1',image1)
            np.save('error2',image2)
            raise ValueError('One of the images contains negative entry!')
        if np.sum(image1) != numerical_weight:
            image1 = image1/w1*numerical_weight if w1 != 0 else np.ones(image1.shape)/np.prod(image1.shape)
        if np.sum(image2) != numerical_weight:
            image2 = image2/w2*numerical_weight  if w2 != 0 else np.ones(image2.shape)/np.prod(image2.shape)
        if np.sum(image1) != numerical_weight or np.sum(image2) != numerical_weight:
            warnings.warn('Could not perfectly normalize: Numerical error occured.')        
    a,b = np.ndarray.flatten(image1), np.ndarray.flatten(image2)
    M = emd_load_M(a,b,image1.shape)
    
    if choice == 'pot_balanced':
        #v = ot.emd2(a,b,M)
        G = ot.emd(a,b,M)
        v = np.sum(G*M)
    elif choice == 'own':
        G,v = emd_dist(a,b,M) #Lame duck    
    elif choice == 'pot_sinkhorn_unbalanced':
        epsilon = 0.05# entropy parameter
        alpha = 100.# Unbalanced KL relaxation parameter
        G = ot_unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha)
        v = np.sum(G*M)
        #    #G[G<0]=0
    G = G/numerical_weight/np.product(image1.shape)*w1 #normalization
    v = v/numerical_weight/np.product(image1.shape)*w1
    return (G,v)

def emd_load_M(a,b,ishape):
    if os.path.isfile('./M_{}x{}.npy'.format(len(a),len(b))):
        M = np.load('./M_{}x{}.npy'.format(len(a),len(b)))        
#        print('Loaded precomputed M')
    else:
        M = np.reshape([emd_euclidian_index_distance(i,j,ishape) for i,j in itertools.product(range(len(a)),range(len(b)))],(len(a),len(b)))
        np.save('./M_{}x{}.npy'.format(len(a),len(b)),M)
        print('Computed M')
    return M    

def emd_euclidian_index_distance(i,j,ishape): #images must have same shape
    indi = np.asarray(np.unravel_index(i, ishape))
    indj = np.asarray(np.unravel_index(j, ishape))
    return np.linalg.norm(indi-indj)

def emd_dist(a,b,M):
    import cvxpy as cp
    verbose_val = True
    
    numerical_weight = len(a)
    a = a/np.sum(a)*numerical_weight
    b = b/np.sum(b)*numerical_weight
    
    G = cp.Variable((len(a),len(b)))
    ones_a = np.ones((1,len(a)))
    a = np.reshape(a,(1,len(a)))
    ones_b = np.ones((len(b),))
    objective = cp.Minimize(cp.sum(cp.multiply(G,M)))
    constraints = [G*ones_b == b, ones_a*G == a,G>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=verbose_val,eps_prim_inf=0)
    return (G.value,np.sum(G.value*M))

def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    if y_hat.shape != y.shape:
        raise ValueError('y_hat and y have different shapes: {} and {}'.format(y_hat.shape,y.shape))
    assert y_hat.shape == y.shape
    return np.mean((y - y_hat) ** 2)


def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        try:
            with open(pkl_filepath, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
        except UnicodeDecodeError:
            with open(pkl_filepath, 'rb') as pkl_file:
                data = pickle.load(pkl_file,encoding='latin1')            
    else:
        print('Pickle not found')
        data = {}
    return data

def get_mode_list():
    return ['flex','alt','fixed']

def get_estimator(hparams, model_type):
    if hparams.dataset in ['mnist','fashion-mnist']:
        if model_type == 'vae':
            estimator = mnist_estimators.vae_estimator(hparams)
        elif np.asarray([mode in model_type for mode in get_mode_list()]).any():
            estimator = mnist_estimators_flexI.flex_estimator(hparams)
#        elif model_type == 'lasso':
#            estimator = mnist_estimators_flexI.lasso_estimator(hparams)
#        elif model_type == 'lasso-wavelet':
#            estimator = mnist_estimators_flexI.lasso_wavelet_estimator(hparams)
#        elif model_type == 'lasso-pca':
#            estimator = mnist_estimators_flexI.lasso_pca_estimator(hparams)
#        elif model_type == 'tv-norm':
#            estimator = mnist_estimators_flexI.tv_estimator(hparams)
        elif model_type in ['lasso','lasso-wavelet','lasso-pca','tv-norm']:
            estimator = mnist_estimators_flexI.lasso_based_estimators(hparams)
        elif model_type in ['dict']:
            estimator = mnist_estimators_flexI.l0_pick()
        elif model_type == 'omp':
            estimator = mnist_estimators_flexI.omp_estimator(hparams)
        elif model_type == 'learned':
            estimator = mnist_estimators.learned_estimator(hparams)
        else:
            print('the type {} is not supported!'.format(model_type))
            raise NotImplementedError
    elif hparams.dataset == 'celebA':
        if model_type == 'lasso-dct':
            estimator = celebA_estimators.lasso_dct_estimator(hparams)
        elif model_type == 'lasso-wavelet':
            estimator = celebA_estimators.lasso_wavelet_estimator(hparams)
        elif model_type == 'lasso-wavelet-ycbcr':
            estimator = celebA_estimators.lasso_wavelet_ycbcr_estimator(hparams)
        elif model_type == 'k-sparse-wavelet':
            estimator = celebA_estimators.k_sparse_wavelet_estimator(hparams)
        elif model_type == 'dcgan':
            estimator = celebA_estimators.dcgan_estimator(hparams)
        elif model_type in ['vae-flex','vae-flex-alt','vae-alt']:
            estimator = mnist_estimators_flexI.vae_estimator_flex(hparams)
        else:
            raise NotImplementedError
    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)


def save_images(est_images, save_image, hparams):
    """Save a batch of images to png files"""
    for model_type in hparams.model_types:
        for image_num, image in est_images[model_type].iteritems():
            save_path = get_save_paths(hparams, image_num)[model_type]
            image = image.reshape(hparams.image_shape)
            save_image(image, save_path)
#            save_path = get_save_paths(hparams, str(image_num)+'nonweird')[model_type]
            
def to_saving_names():
    return {'l2_losses': {},
            'measurement_losses': {},
            'x_orig': [],
            'emd_losses': {},
            'x_rec':[],
            'noise':[],
            'z_rec':[],
            'class_losses' : {}
            }
def loss_names():
    return {
            'l2_losses': {},
            'measurement_losses': {},
            'emd_losses': {},
            'class_losses' : {}
            }

class SaveHandler(object):
    def set_fields(self,fields):
        for fieldname,field in fields.iteritems():
            setattr(self,fieldname,field)
    def get_fields(self):
        return self.__dict__
    def save_all(self,hparams,model_type,restricted=False):
        tmp = 'tmp' if restricted else ''
        path_dict = self.get_pkl_filepaths(hparams,tmp=tmp)[model_type]
        for fieldname,filepath in path_dict.iteritems():
            if not restricted or fieldname in ['x_orig,x_rec']:
                fieldval = getattr(self,fieldname)
                if isinstance(fieldval,dict): #.itervalues().next()
                    save_to_pickle(fieldval[model_type],filepath)
                else:
                    save_to_pickle(fieldval,filepath)
    def load_or_init_all(self,save_images,model_types,path_dicts):
        for field,val in to_saving_names().iteritems():
            if isinstance(val,dict):
                for model_type in model_types:
                    val[model_type] = {}
            setattr(self,field,val)
        if save_images:            
            for model_type in model_types:           
#                path_dict = self.get_pkl_filepaths(hparams, model_type)   
                path_dict = path_dicts[model_type]
                for fieldname,filepath in path_dict.iteritems():
                    fieldval = getattr(self,fieldname)
                    if isinstance(fieldval,dict): #.itervalues().next()
                        fieldval[model_type] = load_if_pickled(filepath)
                    else:
                        fieldval = load_if_pickled(filepath)
                    
    def get_pkl_filepaths(self,hparams,use_all=False,tmp=''):
        """Return paths for the pickle files"""
        paths = {}
        for model_type in hparams.model_types:
            checkpoint_dir = get_checkpoint_dir(hparams, model_type)       
            paths[model_type] = {}
            if use_all:
                for fieldname in to_saving_names():
                    paths[model_type].update({fieldname:checkpoint_dir+fieldname+tmp+'.pkl'})
            else:
                for fieldname,_ in self.get_fields().iteritems():
                    paths[model_type].update({fieldname:checkpoint_dir+fieldname+tmp+'.pkl'})
        return paths

def checkpoint(est_images, save_image, save_handler, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    if hparams.save_images:
        save_images(est_images, save_image, hparams)

    if hparams.save_stats:
        for model_type in hparams.model_types:
            save_handler.save_all(hparams,model_type)
            
def get_classifier_loss(hparams,im1,label):
    import keras
    if label.ndim ==1:  
        im1 = np.reshape(im1,(1,28,28,1))
        label = np.reshape(label,(1,-1))
    if hparams.dataset == 'mnist':
        model = keras.models.load_model('./src/mnist_classifier.h5')
        _, acc = model.evaluate(im1, label, verbose=0)
    elif hparams.dataset == 'fashion-mnist':
        import classifier_fashion
        model = classifier_fashion.create_model()
        model.load_weights('./src/fashionmnist_classifier.hdf5')
        acc = model.evaluate(im1,label, verbose=0)[1]
    elif hparams.dataset == 'celebA':
        model = keras.models.load_model('./src/mnist_classifier.h5')
    
    return 1-acc


def image_matrix(images, est_images, view_image, hparams, alg_labels=True):
    """Display images"""

    if hparams.measurement_type in ['inpaint', 'superres']:
        figure_height = 2 + len(hparams.model_types)
    else:
        figure_height = 1 + len(hparams.model_types)

    fig = plt.figure(figsize=[2*len(images), 2*figure_height])

    outer_counter = 0
    inner_counter = 0

    # Show original images
    outer_counter += 1
    for image in images.values():
        inner_counter += 1
        ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        if alg_labels:
            ax.set_ylabel('Original', fontsize=14)
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)

    # Show original images with inpainting mask
    if hparams.measurement_type == 'inpaint':
        mask = get_inpaint_mask(hparams)
        outer_counter += 1
        for image in images.values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Masked', fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams, mask)

    # Show original images with blurring
    if hparams.measurement_type == 'superres':
        factor = hparams.superres_factor
        A = get_A_superres(hparams)
        outer_counter += 1
        for image in images.values():
            image_low_res = np.matmul(image, A) / np.sqrt(hparams.n_input/(factor**2)) / (factor**2)
            low_res_shape = (int(hparams.image_shape[0]/factor), int(hparams.image_shape[1]/factor), hparams.image_shape[2])
            image_low_res = np.reshape(image_low_res, low_res_shape)
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Blurred', fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image_low_res, hparams)

    for model_type in hparams.model_types:
        outer_counter += 1
        for image in est_images[model_type].values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel(model_type, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
        save_path = get_matrix_save_path(hparams)
        plt.savefig(save_path)

    if hparams.image_matrix in [1, 3]:
        plt.show()


def plot_image(image, cmap=None):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)

def get_all_vaes(path,name):
    name = path+name#+'*'
    print(name)
    vae_list = glob.glob(name+'*')        
    re1 = [re.match(name+'([0-9]*)$',x) for x in vae_list]
    dim_list = [x.group(1) for x in re1 if x != None]
    dim_list = [x for x in dim_list if x !='']
    dim_list = map(int,dim_list)
    dim_list = np.asarray(list(set(dim_list)))
    dim_list.sort()
    return dim_list

def get_checkpoint_dir(hparams, model_type):
    mode_list = get_mode_list()
    n_z = [v if v!=-1 else '' for v in [hparams.n_z]][0]
    if hparams.grid != 'NoGrid':
        grid=process_grid(hparams.grid)    
    else:
        grid=''
    reproducible = '' if hparams.reproducible == '' else '-'+hparams.reproducible
    input_type = hparams.input_type if hparams.input_type != 'random-test' else str(hparams.input_type) + '_' + str(hparams.input_seed) 
    base_dir = './estimated/{0}/{1}/{2}{3}/{4}/{5}/{6}/'.format(
        hparams.dataset,
        input_type,
        hparams.measurement_type,
        reproducible,
        hparams.noise_std,
        hparams.num_measurements,
        ''.join(model_type.split('-')) + str(n_z)+str(grid)
    )

    if model_type in ['lasso', 'lasso-dct','lasso-wavelet-ycbcr']:#,'tv-norm'
        dir_name = '{}_{}'.format(
            hparams.lmbd,
            hparams.lasso_solver
        )
    elif model_type in ['tv-norm','lasso-wavelet','lasso-pca','dict']:
        dir_name = '{}_{}'.format(
            hparams.lmbd,
            hparams.tv_or_lasso_mode+hparams.wavelet_type+hparams.matlab
        )
    elif model_type == 'k-sparse-wavelet':
        dir_name = '{}'.format(
            hparams.sparsity,
        )
    elif model_type in ['vae','vae-alt']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
            hparams.mean,
            hparams.stdv,
            hparams.fair_counter,
            hparams.fair_counter_end
        )
    elif np.asarray([mode in model_type for mode in mode_list]).any():
        eps_vers = str(hparams.tolerance_checking) if 'fixed' not in model_type else ''
        check_vers = str(hparams.strict_checking) if 'alt' in model_type else ''
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
            str(hparams.eps)+eps_vers+check_vers,
            hparams.tol,
            hparams.init_mode,
            hparams.fair_counter,
            hparams.fair_counter_end,
            hparams.flex_chosen
        )
    elif model_type in ['dcgan']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.dloss1_weight,
            hparams.dloss2_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type == 'learned':
        dir_name = '50-200'
    elif model_type == 'omp':
        dir_name = '{}'.format(
            hparams.omp_k)        
    else:
        raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    emd_losses_filepath = checkpoint_dir + 'emd_losses.pkl'
    x_orig_filepath = checkpoint_dir + 'x_orig.pkl'
    x_rec_filepath = checkpoint_dir + 'x_rec.pkl'
    noise_filepath = checkpoint_dir + 'x_noise.pkl'
    return m_losses_filepath, l2_losses_filepath, emd_losses_filepath, x_orig_filepath, x_rec_filepath, noise_filepath


def get_save_paths(hparams, image_num):
    save_paths = {}
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        save_paths[model_type] = checkpoint_dir + '{0}.png'.format(image_num)
    return save_paths


def get_matrix_save_path(hparams):
    save_path = './estimated/{0}/{1}/{2}/{3}/{4}/matrix_{5}.png'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.measurement_type,
        hparams.noise_std,
        hparams.num_measurements,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    print('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')


def get_learning_rate(global_step, hparams):
    if hparams.decay_lr:
        return tf.train.exponential_decay(hparams.learning_rate,
                                          global_step,
                                          50,
                                          0.7,
                                          staircase=True)
    else:
        return tf.constant(hparams.learning_rate)


def get_optimizer(learning_rate, hparams):
    if hparams.optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    if hparams.optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')


def get_inpaint_mask(hparams):
    image_size = hparams.image_shape[0]
    margin = (image_size - hparams.inpaint_size) / 2
    mask = np.ones(hparams.image_shape)
    mask[margin:margin+hparams.inpaint_size, margin:margin+hparams.inpaint_size] = 0
    return mask


def get_A_inpaint(hparams):
    mask = get_inpaint_mask(hparams)
    mask = mask.reshape(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0])

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T


def get_A_superres(hparams):
    factor = hparams.superres_factor
    A = np.zeros((int(hparams.n_input/(factor**2)), hparams.n_input))
    l = 0
    for i in range(hparams.image_shape[0]/factor):
        for j in range(hparams.image_shape[1]/factor):
            for k in range(hparams.image_shape[2]):
                a = np.zeros(hparams.image_shape)
                a[factor*i:factor*(i+1), factor*j:factor*(j+1), k] = 1
                A[l, :] = np.reshape(a, [1, -1])
                l += 1

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input/(factor**2)) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T


def get_A_restore_path(hparams):
    pattern = './optimization/mnist-e2e/checkpoints/adam_0.001_{0}_{1}/'
    if hparams.measurement_type == 'fixed':
        ckpt_dir = pattern.format(hparams.num_measurements, 'False')
    elif hparams.measurement_type == 'learned':
        ckpt_dir = pattern.format(hparams.num_measurements, 'True')
    else:
        raise NotImplementedError
    restore_path = tf.train.latest_checkpoint(ckpt_dir)
    return restore_path


def restore_A(hparams):
    A = tf.get_variable('A', [784, hparams.num_measurements])
    restore_path = get_A_restore_path(hparams)
    model_saver = tf.train.Saver([A])
    with tf.Session() as sess:
        model_saver.restore(sess, restore_path)
        A_val = sess.run(A)
    tf.reset_default_graph()
    return A_val


def get_A(hparams):
    if hparams.measurement_type == 'gaussian':
        if hparams.reproducible == 'reproducible':
            st0 = np.random.get_state()
            np.random.seed(2019)
            A = np.random.randn(hparams.n_input, hparams.num_measurements)
            np.random.set_state(st0)
        elif 'load_' in hparams.reproducible:
            A_path = re.sub('load_', '', hparams.reproducible)
            if A_path == 'standard':
                A = np.load('./data/A_{}.npy'.format(hparams.num_measurements))
            else:
                A = np.load(A_path)
        elif 'standard' == hparams.reproducible:
            raise ValueError('Specify load from standard or load from path!')            
        elif hparams.reproducible == 'None':
            A = np.random.randn(hparams.n_input, hparams.num_measurements)
        else:
            raise ValueError('Mode {} is not defined!'.format(hparams.reproducible))
    elif hparams.measurement_type == 'superres':
        A = get_A_superres(hparams)
    elif hparams.measurement_type in ['fixed', 'learned']:
        A = restore_A(hparams)
    elif hparams.measurement_type == 'inpaint':
        A = get_A_inpaint(hparams)
    elif hparams.measurement_type == 'project':
        A = None
    elif hparams.measurement_type == 'project_it':
        A = np.eye(hparams.n_input)
    elif hparams.measurement_type in ['sample_distribution','sample-distribution','autoencoder']:
        A = np.eye(784)
    else:
        print(hparams.measurement_type)
        raise NotImplementedError
    return A


def set_num_measurements(hparams):
    if hparams.measurement_type == 'project':
        hparams.num_measurements = hparams.n_input
    else:
        hparams.num_measurements = get_A(hparams).shape[1]


def get_checkpoint_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        ckpt_path = ''
    return ckpt_path


def RGB_matrix():
    U_ = np.zeros((12288, 12288))
    U = np.zeros((3, 3))
    V = np.zeros((12288, 1))

    # R, Y
    V[0::3] = ((255.0/219.0)*(-16.0)) + ((255.0*0.701/112.0)*(-128.0))
    U[0, 0] = (255.0/219.0)
    # R, Cb
    U[0, 1] = (0.0)
    # R, Cr
    U[0, 2] = (255.0*0.701/112.0)

    # G, Y
    V[1::3] = ((255.0/219.0)*(-16.0)) - ((0.886*0.114*255.0/(112.0*0.587)) *(-128.0)) - ((255.0*0.701*0.299/(112.0*0.587))*(-128.0))
    U[1, 0] = (255.0/219.0)
    # G, Cb
    U[1, 1] = - (0.886*0.114*255.0/(112.0*0.587))  #*np.eye(4096)
    # G, Cr
    U[1, 2] = - (255.0*0.701*0.299/(112.0*0.587))  #*np.eye(4096)

    # B, Y
    V[2::3] = ((255.0/219.0)*(-16.0)) + ((0.886*255.0/(112.0))*(-128.0))
    U[2, 0] = (255.0/219.0)  #*np.eye(4096)
    # B, Cb
    U[2, 1] = (0.886*255.0/(112.0))  #*np.eye(4096)
    # B, Cr
    U[2, 2] = 0.0

    for i in range(4096):
        U_[i*3:(i+1)*3, i*3:(i+1)*3] = U
    return U_, V


def YCbCr(image):
    """
     input: array with RGB values between 0 and 255
     output: array with YCbCr values between 16 and 235(Y) or 240(Cb, Cr)
    """
    x = image.copy()
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    # Y channel = 16.0 + 65.378/256 R + 129.057/256 * G + 25.064/256.0 * B
    x[:, :, 0] = 16.0 + (65.738/256.0)*R + (129.057/256.0)*G + (25.064/256.0)*B
    # Cb channel = 128.0 - 37.945/256 R - 74.494/256 * G + 112.439/256 * B
    x[:, :, 1] = 128.0 - (37.945/256.0)*R - (74.494/256.0)*G + (112.439/256.0)*B
    # Cr channel = 128.0+ 112.439/256 R - 94.154/256 * G - 18.285/256 * B
    x[:, :, 2] = 128.0 + (112.439/256.0)*R - (94.154/256.0)*G - (18.285/256.0)*B
    return x


def RGB(image):
    """
     input: array with YCbCr values between 16 and 235(Y) or 240(Cb, Cr)
     output: array with RGB values between 0 and 255
    """
    x = image.copy()
    Y = image[:, :, 0]
    Cb = image[:, :, 1]
    Cr = image[:, :, 2]
    x[:, :, 0] = (255.0/219.0)*(Y - 16.0) + (0.0/112.0) *(Cb - 128.0)+ (255.0*0.701/112.0)*(Cr - 128.0)
    x[:, :, 1] = (255.0/219.0)*(Y - 16.0) - (0.886*0.114*255.0/(112.0*0.587)) *(Cb - 128.0) - (255.0*0.701*0.299/(112.0*0.587))*(Cr - 128.0)
    x[:, :, 2] = (255.0/219.0)*(Y - 16.0) + (0.886*255.0/(112.0)) *(Cb - 128.0) + (0.0/112.0)*(Cr - 128.0)
    return x


def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()


def solve_lasso(A_val, y_val, hparams):
    if hparams.lasso_solver == 'sklearn':
        lasso_est = Lasso(alpha=hparams.lmbd)
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [-1])
    elif hparams.lasso_solver == 'sklearn_gs':
        clf1 = FistaRegressor(penalty='l1')
        gs = GridSearchCV(clf1, {'alpha': np.logspace(-3, 3, 10)})
#        print(A_val.shape,y_batch_val.shape)
        x_hat = gs.fit(A_val.T, y_val.T).coef_.ravel()
    elif hparams.lasso_solver == 'cvxopt':
        A_mat = matrix(A_val.T)
        y_mat = matrix(y_val)
        x_hat_mat = l1regls(A_mat, y_mat)
        x_hat = np.asarray(x_hat_mat)
        x_hat = np.reshape(x_hat, [-1])
    return x_hat

def get_opt_reinit_op(opt, var_list, global_step):
    opt_slots = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_slots.extend([opt._beta1_power, opt._beta2_power])  #pylint: disable = W0212
    all_opt_variables = opt_slots + var_list + [global_step]
    opt_reinit_op = tf.variables_initializer(all_opt_variables)
    return opt_reinit_op

def process_grid(grid):
    grid_prep = str(grid)
    grid_prep=grid_prep.replace("[","")
    grid_prep=grid_prep.replace(']','')
    grid_prep=grid_prep.replace(', ','-')
    return grid_prep

#def phase_transition_data(m,n,k):
    
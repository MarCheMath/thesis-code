"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

from sklearn.linear_model import OrthogonalMatchingPursuit
from lightning.regression import FistaRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf

import utils
#import time
#import datetime
import utils_lasso
import utils_vaeflex

#meaningful modes:
#fista
#cvxpy-[synthesis,analysis]-[constr,reg[ ,-param[0-9]*][ ,-reweight[0-9]*]]
#cvxopt-[constr,reg[ ,-param[0-9]*][ ,-reweight[0-9]*]]            
#all lasso-based input_types [lasso-wavelet,lasso,lasso-pca,tv-norm] are admissible, but the analysis option makes only sense for lasso-wavelet. 
#for lasso-wavelet, add in the end [redundant,nonredundant]-[28,64]
def lasso_based_estimators(hparams):  #pylint: disable = W0613
    """Lasso based estimators, includes TV, PCA, Wavelet"""
    def estimator(A_val, y_batch_val, hparams):
        if A_val is None:
            A = np.eye(y_batch_val.shape[1]) #TODO remove idle
        else:
            A = A_val.T
        mode = hparams.tv_or_lasso_mode
        y = utils_lasso.get_y(mode,y_batch_val,A)        
        W = utils_lasso.load_W(hparams, A, y)
        x_hat_batch = []
        if W is None:
            AW = A
        else:
            if W is None or 'tv-norm' in hparams.model_types:
                AW = None           
            else:
                AW = A.dot(W)                
        if 'fista' in mode:
            clf1 = FistaRegressor(penalty='l1')
            gs = GridSearchCV(clf1, {'alpha': np.logspace(-3, 3, 10)})          
            gs.fit(AW, y)
            x_hat_batch = gs.best_estimator_.coef_.dot(W)#.T            
        elif 'cvxpy' in mode:
            maxit = utils_lasso.get_key('reweight',mode,int)
            weight = np.array([])
            errors = []            
            for it in range(maxit):
                if maxit >1:
                    print('Computing reweighting iteration: {}'.format(it+1))
                x_hat_batch,weight = utils_lasso.cvxpy_problem(mode,W,A,AW,y,weight)
                current_error = np.mean(utils.get_l2_loss(np.array(hparams.x),utils_lasso.get_x(mode,x_hat_batch)))#np.mean(np.mean((np.array(hparams.x)-x_hat_batch)**2,axis=1))
                errors.append(current_error)
                print('Current error: {}'.format(current_error))
            if maxit>1:
                errors = np.array(errors)
                np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'error_course',errors)            
        elif 'cvxopt' in mode:
            for i in range(hparams.batch_size):
                y_val = y_batch_val[i]
                x_hat = utils_lasso.tv_cvxopt_partial(A_val, y_val,hparams)
                x_hat_batch.append(x_hat)
                print('Processed {} percent of all images'.format(float(i+1)/hparams.batch_size*100))
            x_hat_batch = np.asarray(x_hat_batch)
        elif 'sklearn' in mode:            
            for i in range(hparams.batch_size):
                y_val = y_batch_val[i]
                x_hat = utils.solve_lasso(A_val, y_val, hparams)
                x_hat = np.maximum(np.minimum(x_hat, 1), 0)
                x_hat_batch.append(x_hat)
                print('Processed {} percent of all images'.format(float(i+1)/hparams.batch_size*100))
            x_hat_batch = np.asarray(x_hat_batch)
        elif 'bestkTermApprox' in mode:
            for i in range(hparams.batch_size):
                y_val = y_batch_val[i]
                x_hat = utils_lasso.check_bestkTermApprox(mode, W, y_val)
                x_hat = np.maximum(np.minimum(x_hat, 1), 0)
                x_hat_batch.append(x_hat)
                print('Processed {} percent of all images'.format(float(i+1)/hparams.batch_size*100))
            x_hat_batch = np.asarray(x_hat_batch)
        elif 'bruteForce' in mode: #slithly redundant to above
            for i in range(hparams.batch_size):
                y_val = y_batch_val[i]
                x_hat = utils_lasso.brute_force(mode, A_val,W,y_val)
                x_hat = np.maximum(np.minimum(x_hat, 1), 0)
                x_hat_batch.append(x_hat)
                print('Processed {} percent of all images'.format(float(i+1)/hparams.batch_size*100))
            x_hat_batch = np.asarray(x_hat_batch)
        else:
            raise NotImplementedError('Mode {} does not exist!'.format(mode))
        return utils_lasso.get_x(mode,x_hat_batch)    
    return estimator                   
#2,2,4,4,4,16,16,16,49,49,49,196,196,196
#392,392,196,196,196,49,49,49,16,16,16,4,4,4
#(19.78x19.78),(14x14),(7x7),(4x4),(2x2)
    
def l0_pick():
    def estimator(A_val, y_batch_val, hparams):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
        rawDictLength = utils_lasso.get_key('rawDictLength',hparams.tv_or_lasso_mode,int,standard=784)
        W = mnist.train.images.T[:,:rawDictLength]
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            x_hat = utils_lasso.brute_force(hparams.tv_or_lasso_mode, A_val.T,W,y_val)
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
            print('Processed {} percent of all images'.format(float(i+1)/hparams.batch_size*100))
        return np.asarray(x_hat_batch)
    return estimator
            
            
def omp_estimator(hparams):
    """OMP estimator"""
    omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs=hparams.omp_k)
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = utils_lasso.load_W(hparams,A_val,y_batch_val);
        if A_val is None:
            AW=W
        else:
            AW = A_val.T.dot(W)
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            omp_est.fit(AW, y_val.reshape(hparams.num_measurements))
            x_hat = omp_est.coef_
            x_hat = np.reshape(x_hat, [-1])
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator

def flex_estimator(hparams):   
    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        utils_vaeflex.set_model(hparams)
        recovered=False
        if hparams.recovered:
            print('----------Old optimization found, resuming it!----------')
            bs = hparams.optim.bs
            bs.recovered = True
            recovered = True
            init_obj = hparams.optim.init_obj            
        else:
            if ('alt' in hparams.model_types[0] and hparams.strict_checking=='strict'):
                bs_param = utils_vaeflex.get_dimlist(hparams)
            elif ('flex' in hparams.model_types[0] and hparams.strict_checking=='strict'):
                bs_param = hparams.grid
            else:
                bs_param = 'nonstrict'
            bs = utils_vaeflex.Binary_search(hparams.grid[-1],hparams.tol,hparams.flex_chosen,strict_grid=bs_param,fair_counter=hparams.fair_counter)
            init_obj = utils_vaeflex.initObject(hparams.init_mode,hparams.num_random_restarts,init_init_mode='random')
        x_hat = []
        z_hat = 'This should be overwritten, otherwhise the procedure has the wrong setup'
        
        print('The initial mode is: {}'.format(init_obj.init_init_mode))
        print('The regular mode is: {}'.format(init_obj.init_mode))#legit printer
        unfair_counter = 0
        for i in bs:
            unfair_counter = unfair_counter + 1
            print('Hidden dimension is {}'.format(i))
            if not recovered:
                utils_vaeflex.setup_for_stage(init_obj,bs,z_hat)
            flex_bol = hparams.flex_chosen=='flexible'
            x_hat, z_hat = utils_vaeflex.stage_i(A_val,y_batch_val,hparams,i,init_obj,flex_bol,bs,hparams.optim,recovered=recovered)
            recovered=False
            mean_dist_pix = utils_vaeflex.check_tolerance(hparams,A_val,x_hat,y_batch_val)
            print('Mean pix distance is {}'.format(mean_dist_pix[0]))
            bs.set_criterion(mean_dist_pix[1],x_hat,z_hat)
            init_obj.init_done()
        if not mean_dist_pix[1]: #mean_dist_pix[0]>=hparams.eps:
            x_hat = bs.up_val
            final_i = bs.j
        else:
            final_i = i
        if flex_bol:#measurement loss could be none of the previous, since selecting for each image the best, not the joint mean.
            print('---------------The final chosen hidden dimension is {}------------------'.format(final_i))
            np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'chosen_dim',final_i)
            np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'fair_counter',unfair_counter + hparams.fair_counter_end)
            for _ in range(hparams.fair_counter_end):
                utils_vaeflex.setup_for_stage(init_obj,bs,z_hat)
                x_hat,z_hat = utils_vaeflex.stage_i(A_val,y_batch_val,hparams,final_i,init_obj,False,bs,hparams.optim,recovered=recovered)#change_return z_hat
#        np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'elapsed_time',duration)
        return (x_hat,z_hat)        
    return estimator



#--------------------------------Think before removing---------------------------------
#def learned_estimator(hparams):
#
#    sess = tf.Session()
#    y_batch, x_hat_batch, restore_dict = mnist_model_def.end_to_end(hparams)
#    restore_path = utils.get_A_restore_path(hparams)
#
#    # Intialize and restore model parameters
#    restorer = tf.train.Saver(var_list=restore_dict)
#    restorer.restore(sess, restore_path)
#
#    def estimator(A_val, y_batch_val, hparams):  # pylint: disable = W0613
#        """Function that returns the estimated image"""
#        x_hat_batch_val = sess.run(x_hat_batch, feed_dict={y_batch: y_batch_val})
#        return x_hat_batch_val
#
#    return estimator

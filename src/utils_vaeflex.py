#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:17:01 2019

@author: marche
"""

import numpy as np
import tensorflow as tf
#from mnist_vae_flex.src import model_def
import generative_model_collections.flex_wrapper_DIEHARD as wrapper
import utils
from mnist_utils import save_image
import copy

def get_module_name(hparams):
    model_type = hparams.model_types[0]
    model_type = model_type.split('-')
    model_type = list(set(model_type).difference(set(utils.get_mode_list())))[0]
    return model_type

def set_model(hparams):
    model_type = get_module_name(hparams)
    print('Using {} as model'.format(model_type))
    #Model has to have the functions generator_i, get_z_var, gen_restore_vars and the class Hparams (look e.g. for vaeflex)
    if np.asarray([mode in model_type for mode in ['vae']]).any():
        globals()['model_def'] = __import__('mnist_vae_flex.src', fromlist=['model_def']).model_def
    #elif np.asarray([mode in model_type for mode in ['BEGAN']]).any():
    else:
        globals()['model_def'] = wrapper.init(model_type,training=False)(sess='only mode def',epoch='only mode def',batch_size=hparams.batch_size,dataset_name=hparams.dataset,checkpoint_dir='./generative_model_collections/checkpoint',result_dir='./generative_model_collections/results',log_dir='./generative_model_collections/logs',grid=hparams.grid,repetition_bol=hparams.repetition_bol) #returns instance of class    

def stage_i(A_val,y_batch_val,hparams,hid_i,init_obj,early_stop,bs,optim,recovered=False):
    model_def = globals()['model_def']
    m_loss1_batch_dict = {}
    m_loss2_batch_dict = {}
    zp_loss_batch_dict = {}
    total_loss_dict = {}
    x_hat_batch_dict = {}
    model_selection = ModelSelect(hparams) 
    hid_i=int(hid_i)
#        print('Matrix norm is {}'.format(np.linalg.norm(A_val)))
#        hparams.eps = hparams.eps * np.linalg.norm(A_val)
   
    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
   
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')
    # Create the generator
    model_hparams = model_def.Hparams()
    model_hparams.n_z = hparams.n_z
    model_hparams.stdv = hparams.stdv
    model_hparams.mean = hparams.mean
    model_hparams.grid = copy.deepcopy(hparams.grid)
    model_selection.setup_dim(hid_i,model_hparams)
    
    if not hparams.model_types[0] == 'vae-flex-alt' and 'alt' in hparams.model_types[0]:
        model_def.ignore_grid = next((j for  j in model_selection.dim_list if j >= hid_i), None)
    
    #set up the initialization            
    print('The initialization is: {}'.format(init_obj.mode))
    if init_obj.mode=='random':
        z_batch = model_def.get_z_var(model_hparams,hparams.batch_size,hid_i)
    elif init_obj.mode in ['previous-and-random','only-previous']:
        z_batch = model_def.get_z_var(model_hparams,hparams.batch_size,hid_i)
        init_op_par = tf.assign(z_batch, truncate_val(model_hparams,hparams,hid_i,init_obj,stdv=0))
    else:
        z_batch = truncate_val(model_hparams,hparams,hid_i,init_obj,stdv=0.1)
    _, x_hat_batch, _ = model_def.generator_i(model_hparams, z_batch, 'gen', hparams.bol,hid_i,relative=False)
    x_hat_batch_dict[hid_i] = x_hat_batch


    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    
    if hparams.stdv>0:
        norm_val = 1/(hparams.stdv**2)
    else:
        norm_val = 1e+20
    
    zp_loss_batch = tf.reduce_sum((z_batch-tf.ones(tf.shape(z_batch))*hparams.mean)**2*norm_val, 1) #added normalization       
    
    # define total loss    
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)
    total_loss_dict[hid_i] = total_loss
    
    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    
    m_loss1_batch_dict[hid_i] = m_loss1
    m_loss2_batch_dict[hid_i] = m_loss2
    zp_loss_batch_dict[hid_i] = zp_loss

    # Set up gradient descent
    var_list = [z_batch]
    if recovered:
        global_step = tf.Variable(hparams.optim.global_step, trainable=False, name='global_step')
    else:
        global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #restore the setting
    if 'alt' in hparams.model_types[0]:
        factor = 1
    else:
        factor = len(hparams.grid)
    model_def.batch_size = hparams.batch_size*factor #changes object (call by reference), necessary, since call of generator_i might change batch size.
    model_selection.restore(sess,hid_i)        

    if recovered:
        best_keeper = hparams.optim.best_keeper
    else:
        best_keeper = utils.BestKeeper(hparams,logg_z=True)
    if hparams.measurement_type == 'project':
        feed_dict = {y_batch: y_batch_val}
    else:
        feed_dict = {A: A_val, y_batch: y_batch_val}
    flag = False
    for i in range(init_obj.num_random_restarts):
        if recovered and i <= hparams.optim.i: #Loosing optimizer's state, keras implementation maybe better
            if i < hparams.optim.i:
                continue
            else:
                sess.run(utils.get_opt_reinit_op(opt, [], global_step))
                sess.run(tf.assign(z_batch,hparams.optim.z_batch))              
        else:            
            sess.run(opt_reinit_op)
            if i<1 and init_obj.mode in ['previous-and-random','only-previous']:
                print('Using previous outcome as starting point')
                sess.run(init_op_par)            
        for j in range(hparams.max_update_iter):
            if recovered and j < hparams.optim.j:
                continue
            _, lr_val, total_loss_val, \
            m_loss1_val, \
            m_loss2_val, \
            zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                    m_loss1,
                                    m_loss2,
                                    zp_loss], feed_dict=feed_dict)         

            if hparams.gif and ((j % hparams.gif_iter) == 0):
                images = sess.run(x_hat_batch, feed_dict=feed_dict)
                for im_num, image in enumerate(images):
                    save_dir = '{0}/{1}/{2}/'.format(hparams.gif_dir, hid_i,im_num)
                    utils.set_up_dir(save_dir)
                    save_path = save_dir + '{0}.png'.format(j)
                    image = image.reshape(hparams.image_shape)
                    save_image(image, save_path)
            if j%100==0 and early_stop:
                x_hat_batch_val = sess.run(x_hat_batch, feed_dict=feed_dict)
                if check_tolerance(hparams,A_val,x_hat_batch_val,y_batch_val)[1]:
                    flag = True
                    print('Early stopping')
                    break
            if j%25==0:#Now not every turn                
                logging_format = 'hid {} rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print( logging_format.format(hid_i, i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val)) 
            if j%100==0:
                x_hat_batch_val, total_loss_batch_val, z_batch_val = sess.run([x_hat_batch, total_loss_batch,z_batch], feed_dict=feed_dict)
                best_keeper.report(x_hat_batch_val, total_loss_batch_val,z_val=z_batch_val)
                optim.global_step = sess.run(global_step)
                optim.A = A_val
                optim.y_batch = y_batch_val
                optim.i=i
                optim.j=j
                optim.z_batch= z_batch_val
                optim.best_keeper=best_keeper
                optim.bs=bs
                optim.init_obj = init_obj
                utils.save_to_pickle(optim,utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'tmp/optim.pkl')
                print('Checkpoint of optimization created')

        hparams.optim.j = 0                
        x_hat_batch_val, total_loss_batch_val, z_batch_val = sess.run([x_hat_batch, total_loss_batch,z_batch], feed_dict=feed_dict)
        best_keeper.report(x_hat_batch_val, total_loss_batch_val,z_val=z_batch_val)
        if flag:
            break
    tf.reset_default_graph()
    return best_keeper.get_best()

class Optim(object):
    pass

def setup_for_stage(init_obj,bs,z_hat):
    if init_obj.mode=='random':                
        pass
    elif 'reuse-next-larger' in init_obj.mode:
        init_obj.set_val(bs.z_up_val)                
    elif 'reuse-last-stage' in init_obj.mode:
        init_obj.set_val(z_hat)
    elif 'previous-and-random' in init_obj.mode:
        init_obj.set_val(z_hat)                        
    elif 'only-previous' in init_obj.mode:
        init_obj.set_val(z_hat)
        init_obj.num_random_restarts = 1
    else:
        raise NotImplementedError('The mode {} is not implemented'.format(init_obj.mode))

def truncate_val(model_hparams,hparams,hid_i,init_obj,stdv=0.1):
    model_def = globals()['model_def']
    #If perturbation wanted, the value of the initialization is augmented by noise
    if 'perturbation' in init_obj.mode:
        print('Perturbation added with strength {} as variance'.format(stdv))
        pre_init = model_def.get_z_var(model_hparams,hparams.batch_size,thick=hid_i,stdv=stdv,mean=np.float32(init_obj.val)) #tf.random gives different results than np.random?
    else:
        pre_init = np.float32(init_obj.val)
    
    #tailor shape of the initialization such that it fits to required hidden dimension
    if pre_init.shape[1]<=hid_i:
        tmp = tf.cast(model_def.get_z_var(model_hparams,hparams.batch_size,thick=hid_i-pre_init.shape[1],stdv=hparams.stdv,mean=np.float32(hparams.mean)),tf.float32)#tf.random_normal(((hparams.batch_size,hid_i-pre_init.shape[1])),stddev=hparams.stdv,mean=hparams.mean,dtype=tf.float32)#attention GANS (uniform distribution) 
        pre_init = tf.concat([pre_init,tmp],axis=1)
        z_batch = tf.get_variable('z',dtype=tf.float32,initializer=pre_init)
    else:                
        z_batch = tf.get_variable('z',dtype=tf.float32,initializer=tf.slice(pre_init,(0,0),(pre_init.shape[0],hid_i)))
    hparams.debug_pi = pre_init
    hparams.debug_z_batch = z_batch
    return z_batch
        
def check_tolerance(hparams,A_val,x_hat,y_batch_val):
    #check whether the distance between y_hat and y is smaller than eps
    if hparams.tolerance_checking == 'non-squared':
        if hparams.measurement_type == 'project':
            dist_y_yhat = np.sqrt(np.sum((np.array(x_hat)-y_batch_val)**2,axis=1))
        else:
            dist_y_yhat = np.sqrt(np.sum((np.dot(np.array(x_hat),np.array(A_val))-y_batch_val)**2,axis=1))
    elif hparams.tolerance_checking == 'squared':
        dist_y_yhat = utils.get_measurement_loss(x_hat, A_val, y_batch_val)
    else:
        raise NotImplementedError('Tolerance-checking mode {} does not exist'.format(hparams.tolerance_checking))
    mean_dist = np.mean(dist_y_yhat)
    mean_dist_pix = mean_dist
#    mean_dist_pix = mean_dist/hparams.n_input#np.asarray(x_hat).shape[1]
    if hparams.measurement_type == 'project':
        chk_tol = False #When projecting, the measurement loss is the representation loss, there is no reason to impose a bias. Just use the highest dimension.
    else:
        chk_tol = mean_dist_pix<hparams.eps    
    print '------------>meandistpix:{}'.format(mean_dist_pix)
    return mean_dist_pix, chk_tol
#    return mean_dist, chk_tol

def get_dimlist(hparams):
    model_type = hparams.model_types[0]
    base_path = hparams.pretrained_model_dir
    if model_type in ['vae-flex-alt','vae-flex']:
        dim_list = utils.get_all_vaes(base_path,'mnist-vae-flex-')
    elif model_type in ['vae-alt']:
        dim_list = utils.get_all_vaes(base_path,'mnist-vae')
    elif np.asarray([mode in model_type for mode in utils.get_mode_list()]).any():
        if 'alt' in model_type:
            factor = 1
        else:
            factor = len(hparams.grid)
        dim_list = utils.get_all_vaes(base_path,'{}/{}_{}_{}_{}_'.format(get_module_name(hparams),get_module_name(hparams),hparams.dataset,hparams.batch_size*factor,hparams.repetition_bol)) #base path more basy
    else:
        raise NotImplementedError('For this file, the input type {} has not been implemented'.format(model_type))
    print('The used dimensions list is: {}'.format(dim_list))       
    return dim_list

class initObject:
    #init_mode has the form mode or, if perturbation desired, mode_perturbation.
    #mode is random, reuse_next_larger or reuse_last_stage
    def __init__(self,init_mode,num_random_restarts,init_init_mode = 'random'):
        self.mode=init_init_mode
        self.init_mode=init_mode        
        self.init_init_mode=init_init_mode
        self.num_random_restarts=num_random_restarts
        if 'perturbation' in init_mode:
            self.perturbation = True
        else:
            self.perturbation = False        
            
    def set_val(self,val):
        self.val=val
    def init_done(self):
        self.mode = self.init_mode
            

class Binary_search:
    def __init__(self,n,tol,flex_chosen,strict_grid='nonstrict',fair_counter='unequal'):
        self.recovered=False
        self.n=n        
        if isinstance(strict_grid,str) and strict_grid == 'nonstrict':
            self.strict_grid=strict_grid
            if flex_chosen != 'flexible':
                self.i=int(flex_chosen)
                self.j=int(flex_chosen)                
            else:
                self.i=1
                self.j=self.n
        else:
            self.strict_grid=sorted(strict_grid)
            self.i=np.min(self.strict_grid)
            self.j=np.max(self.strict_grid)            
        self.flex_chosen = flex_chosen
        self.pointer = self.j
        self.criterion = 'init1'
        self.low_val = []
        self.up_val = []
        self.z_up_val = 'no_pre_init'
        self.tol=tol
        self.fair=0
        self.fair_counter = fair_counter
        
    def __iter__(self):
        return self
    def next(self):        
        self.fair = self.fair+1
        if self.recovered:
            self.recovered = False
            return self.pointer
        if self.criterion=='StopIteration':
            raise StopIteration
        elif self.criterion=='init1':
            if self.flex_chosen == 'flexible':
                self.pointer = self.j#n
            else:
                self.pointer = int(self.flex_chosen)
        elif self.criterion=='init2':
            self.pointer = self.i#1
        elif self.criterion == 'met':       
            self.j = self.pointer
            tmp = self.pointer
            self.pointer = np.trunc(self.i+(self.j-self.i)/2)
            if not self.strict_grid == 'nonstrict':
                self.pointer = self.strict_grid[np.argmin(np.abs(np.asarray(self.strict_grid)-self.pointer))]        
#                self.pointer = next((j for  j in self.strict_grid if j >= self.pointer), None)
                if tmp ==self.pointer: #One could also perform binary search on the indices of the dimension list, but makes less sense to me.
#                    self.pointer = next((j for  j in sorted(self.strict_grid,reverse=True) if j < self.pointer), None)
                    self.pointer = next((j for  j in self.strict_grid if j >= self.pointer), None)
        elif self.criterion == 'not_met':
            self.i = self.pointer
            tmp = self.pointer
            self.pointer = np.trunc(self.i+(self.j-self.i)/2)
            if not self.strict_grid == 'nonstrict':
                self.pointer = self.strict_grid[np.argmin(np.abs(np.asarray(self.strict_grid)-self.pointer))]        
                if tmp ==self.pointer: #One could also perform binary search on the indices of the dimension list, but makes less sense to me.
                    self.pointer = next((j for  j in self.strict_grid if j >= self.pointer), None)
        else:
            raise ValueError
        if (not self.criterion == 'init1') and (self.j-self.i <=np.maximum(1,self.tol) or ((not self.strict_grid == 'nonstrict') and self.j==next((j for  j in self.strict_grid if j > self.i), None)) ) and (self.fair_counter == 'unequal' or self.fair>int(self.fair_counter)):
            raise StopIteration            
        else:
            return self.pointer           
    def set_criterion(self,in_B,val,z_val):
        if ((self.criterion=='init1' and not in_B) or self.flex_chosen!='flexible') and (self.fair_counter == 'unequal' or self.fair>=int(self.fair_counter)):
            criterion = 'StopIteration'
        elif self.criterion=='init1' and in_B:
            criterion='init2'
        elif in_B:
            criterion = 'met'
        else:
            criterion = 'not_met'
        if self.criterion in ['init1','met']:
            self.up_val = val
            self.z_up_val = z_val
        elif self.criterion in ['init2','not_met']:
            self.low_val = val
        else:
            raise ValueError            
        self.criterion = criterion      

class ModelSelect(object):
    def __init__(self,hparams):
        self.base_path = hparams.pretrained_model_dir
        self.model_type = hparams.model_types[0]
        self.dim_list = get_dimlist(hparams)
        self.current_dim = hparams.grid[-1]
    def restore(self,sess,hid_i):
        model_def = globals()['model_def']
        if str(type(model_def))==str(wrapper.init(self.model_type.split('-')[0],training=False)):#isinstance(model_def,wrapper.init(self.model_type.split('-')[0],training=False)):
            model_def.saver = tf.train.Saver()
            print('base path is {}'.format(self.base_path))
            if 'GAN' in str.upper(self.model_type):
                var_list = {var.op.name:var for var in tf.global_variables() if 'generator' in var.op.name or 'g_' in var.op.name}
            else:
                var_list = {var.op.name:var for var in tf.global_variables() if 'decoder' in var.op.name or 'de_' in var.op.name}
            if len(var_list)==0:    #Don't be 'pythonic'!
                raise ValueError('The list of variables to load is empty {}, check scope and names! General list is {}'.format(var_list,{var.op.name:var for var in tf.global_variables()}))
#            print 'VARIABLE LIST {}'.format(var_list)
#            path = self.base_path
#            print path
#            restore_path = tf.train.latest_checkpoint(path)
#            restorer = tf.train.Saver(var_list=var_list)
#            restorer.restore(sess, restore_path)
            ignore_grid = next((j for  j in self.dim_list if j >= hid_i), None) if 'alt' in self.model_type else 'obey_grid'
            ret = model_def.load(self.base_path,sess,var_list,ignore_grid)
            if ret is not None and len(ret)==2:
                if not ret[0]:
                    import warnings
                    warnings.warn('Model probably not loaded!')
        else:
            if self.model_type == 'vae-flex-alt':
                selected_dim = next((j for  j in self.dim_list if j >= hid_i), None)                      
                restore_vars = model_def.gen_restore_vars()
                restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}            
                path = self.base_path+'mnist-vae-flex-'+str(selected_dim)
                print(path)
                restore_path = tf.train.latest_checkpoint(path)
                restorer = tf.train.Saver(var_list=restore_dict)
                restorer.restore(sess, restore_path)
            elif self.model_type == 'vae-alt':
                selected_dim = next((j for  j in self.dim_list if j >= hid_i), None)                      
                restore_vars = model_def.gen_restore_vars()
                restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}            
                path = self.base_path+'mnist-vae'+str(selected_dim)
                print(path)
                restore_path = tf.train.latest_checkpoint(path)
                restorer = tf.train.Saver(var_list=restore_dict)
                restorer.restore(sess, restore_path)
            else:
                restore_vars = model_def.gen_restore_vars()
                restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
                print(self.base_path)
                restore_path = tf.train.latest_checkpoint(self.base_path)
                restorer = tf.train.Saver(var_list=restore_dict)
                restorer.restore(sess, restore_path)
    def setup_dim(self,hid_i,model_hparams):
        if 'alt' in self.model_type:
            selected_dim = next((j for  j in self.dim_list if j >= hid_i), None)   
            model_hparams.grid[-1] = selected_dim
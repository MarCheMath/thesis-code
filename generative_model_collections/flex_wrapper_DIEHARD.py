#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:34:12 2019

@author: marche
"""
import warnings
import os

def init(name,training=True):
    if name == 'vae':
        return 'STANDARD'
    try:
        if training:
            model = getattr(__import__(name), name)
        else:
            model = getattr(__import__('generative_model_collections.{}'.format(name), fromlist=[name]),name)       
    except:
        warnings.warn('Module not found, using the standard vae now!')
        return 'STANDARD'
    import numpy as np
    import itertools
    import tensorflow as tf

    class flex_wrapper(model):
        class Hparams(object):
            pass
        def __init__(self, *args,**kwargs):
            grid = kwargs.pop('grid')
            if 'repetition_bol' in kwargs:
                repetition_bol = kwargs.pop('repetition_bol')
            else:
                repetition_bol = 'False'
            if 'GAN' not in name.upper():
                repetition_bol = 'False'
            self.repetition_bol = repetition_bol
            self.grid = np.sort(grid)
            self.z_dim = np.max(grid)
            self.batch_size_fraction = len(grid)
            kwargs.update({'z_dim':self.z_dim})
            super(flex_wrapper,self).__init__(*args,**kwargs)   
            try:
                self.data_X = list(itertools.chain.from_iterable(itertools.repeat(x, self.batch_size_fraction) for x in self.data_X))
                self.data_y = list(itertools.chain.from_iterable(itertools.repeat(x, self.batch_size_fraction) for x in self.data_y))
            except AttributeError:
                self.data = list(itertools.chain.from_iterable(itertools.repeat(x, self.batch_size_fraction) for x in self.data))
                warnings.warn('LOOK AT ROLE OF z_dim!')     
            self.true_batch_size = self.batch_size                              
            if repetition_bol == 'True':
                self.repetition = len(grid)
            else:
                self.repetition = 1
            if 'VAE' in name:
                self.batch_size = int(self.batch_size * self.batch_size_fraction)
        if 'VAE' in name:            
            def encoder(self, *args, **kwargs):#KL divergence will compare 0 to normal, but this is just a constant, which won't change the minimizer
                if 'reuse' in kwargs:
                    reuse = kwargs['reuse']
                else:
                    reuse = False
                if 'is_training' in kwargs:
                    is_training = kwargs['is_training']
                else:
                    is_training = True
                self.batch_size = self.true_batch_size
                x = args[0]
                x =  [tf.slice(x,(i,0,0,0),(1,self.input_height, self.input_width, self.c_dim)) for i in (np.asarray(range(self.true_batch_size))*self.batch_size_fraction)]            
                x = tf.concat(x,0)
                if len(args)>1:
                    y = args[1]
                    y =  [tf.slice(y,(i,0),(1,self.y_dim)) for i in (np.asarray(range(self.true_batch_size))*self.batch_size_fraction)]            
                    y = tf.concat(y,0)
                    mean,stdv = super(flex_wrapper,self).encoder(x, y, is_training=is_training, reuse=reuse)
                else:
                    mean,stdv = super(flex_wrapper,self).encoder(x, is_training=is_training, reuse=reuse)
                                   
                self.batch_size = int(self.batch_size * self.batch_size_fraction)
                return self.split_and_merge(mean),self.split_and_merge(stdv)
 
            def generator_i(self,dummy1,z,dummy2,dummy3,hid_i,**kwargs):
                if hasattr(self,'ignore_grid') and self.ignore_grid != 'obey_grid':
                    exp_siz = self.ignore_grid
                else:
                    exp_siz = self.z_dim
                z_hid_i = slicer_enc(z,end=hid_i,expected_size=exp_siz,batchend=self.true_batch_size)
                self.batch_size = int(z.get_shape()[0])
                try:
                    if hasattr(self,'y'):
                        out = super(flex_wrapper,self).decoder(z_hid_i,self.y,is_training=False,reuse=False)    
                    else:
                        out = super(flex_wrapper,self).decoder(z_hid_i,is_training=False,reuse=False)
                except ValueError:
                    if hasattr(self,'y'):
                        out = super(flex_wrapper,self).decoder(z_hid_i,self.y,is_training=False,reuse=True)    
                    else:
                        out = super(flex_wrapper,self).decoder(z_hid_i,is_training=False,reuse=True)
                out = tf.reshape(out,(int(out.get_shape()[0]),-1))
                return ('',out,'')
        elif 'GAN' in name:
            def generator(self, z, is_training=True, reuse=False):
                self.batch_size = int(self.batch_size * self.batch_size_fraction)
                z_large = self.split_and_merge(z)                
                return_val = super(flex_wrapper,self).generator(z_large, is_training=is_training, reuse=reuse)
                self.batch_size = self.true_batch_size
                return return_val
                
            def discriminator(self,x, is_training=True, reuse=False):
                if x.shape[0] == self.true_batch_size:
                    return super(flex_wrapper,self).discriminator(x,is_training=is_training,reuse=reuse) #already normalized by batch_size
#                    return out, tf.tile(recon_error,self.batch_size_fraction), code #instead of computing error multiple times. Careful with dependence to final error function
                elif x.shape[0] == self.true_batch_size*self.batch_size_fraction:
                    self.batch_size = int(self.batch_size * self.batch_size_fraction)
                    return_cal = super(flex_wrapper,self).discriminator(x,is_training=is_training,reuse=reuse) 
                    self.batch_size = self.true_batch_size
                    return return_cal                    
                else:
                    raise ValueError('Batch size neither real one nor the one with differently sliced z')

            def generator_i(self,dummy1,z,dummy2,dummy3,hid_i,**kwargs):
                z_hid_i = slicer_enc(z,end=hid_i,expected_size=self.z_dim)
                try:
                    out = super(flex_wrapper,self).generator(z_hid_i,is_training=False,reuse=False)    
                except ValueError:
                    out = super(flex_wrapper,self).generator(z_hid_i,is_training=False,reuse=True)
                out = tf.reshape(out,(int(out.get_shape()[0]),-1))     #if multiple channels, packed together (order: channel1 (w*h array), channel2 (w*h array),... )
                return ('',out,'')
        def get_z_var(self,hparams, batch_size, thick = 'notdefined', stdv = 'notdefined', mean = 'notdefined'):
            if thick == 'notdefined':
                thick = hparams.grid[-1]
            if stdv == 'notdefined':
                stdv = hparams.stdv
            if mean == 'notdefined':
                mean = hparams.mean
            if name.upper() in ['VAE']:
                return_val = tf.Variable(tf.random_normal((batch_size, thick),stddev=stdv,mean=mean), name='z')
            else:
                return_val = tf.Variable(tf.random_uniform((batch_size , thick),minval=-1,maxval=1), name='z')
            return return_val
        
        def split_and_merge(self, z):
            z_list = []
            for i in range(len(self.grid)):
                z_list.append(slicer_enc(z,end=self.grid[i],expected_size=self.z_dim,batchend=self.true_batch_size))
            all_z = tf.concat(z_list,0)
            swap_ind = [i*self.true_batch_size+j  for j in range(self.true_batch_size) for i in range(self.batch_size_fraction)]
            all_z = tf.concat([tf.slice(all_z,(i,0),(1,tf.shape(all_z)[1])) for i in swap_ind],0)
            all_z = tf.reshape(all_z,[self.batch_size,self.z_dim])
            #all_z = tf.reshape(all_z,[self.true_batch_size*self.batch_size_fraction,self.z_dim])
            return all_z
        
        @property
        def model_dir(self):
            if not hasattr(self,'ignore_grid') or self.ignore_grid == 'obey_grid':
                import re
                if not isinstance(self.grid,str): #very unnice, but too many different functions with different inputs
                    grid_str = str(self.grid).replace('[','').replace(']','').replace(' ','-')
                    grid_str = str.replace(grid_str,'--','-')
                    grid_str = grid_str if grid_str[0]!='-' else grid_str[1:]
                else:
                    grid_str = str(self.grid)
                    grid_str = '  '.join(grid_str.split(',')).replace('[','').replace(']','')
                    matches = re.finditer(r' (\d+)$',grid_str)
                    results = [int(match.group(1)) for match in matches]
                    matches = re.finditer(r' (\d+) ',grid_str)
                    results = results+ [int(match.group(1)) for match in matches]
                    matches = re.finditer(r'^(\d+) ',grid_str)
                    results = results + [int(match.group(1)) for match in matches]
                    grid_str = '-'.join([str(x) for x in sorted(list(set(results)))])
                    grid_str = str.replace(grid_str,'--','-')
                return "{}_{}_{}_{}_{}".format(
                    name,self.dataset_name,
                    self.batch_size, self.repetition_bol,
                    grid_str
                    )              
            else:
                return "{}_{}_{}_{}_{}".format(
                    name,self.dataset_name,
                    self.batch_size, self.repetition_bol,
                    self.ignore_grid)  
                
        def save(self, checkpoint_dir, step):
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)
    
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
    
            self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

        def load(self, *args):
            if len(args)>1:                
                checkpoint_dir,sess,var_list = args[0],args[1],args[2]
                saver = tf.train.Saver(var_list=var_list)
            else:
                checkpoint_dir,sess = args[0],self.sess
                saver = self.saver
            if len(args)== 4:
                self.ignore_grid = args[3]
            else:
                self.ignore_grid = 'obey_grid'
            import re
            print(" [*] Reading checkpoints...")
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)
            print(' [*] Searching at {}'.format(checkpoint_dir))   
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)              
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                print counter
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0

            
    def slicer_enc(v, start = 0, end = 'nothing',expected_size = 'size of v',batchstart = 0, batchend = 'nothing'):
        if batchend == 'nothing':
            batchend = int(v.get_shape()[0])#batch size, at construction time often undefined!!!
        if end == 'nothing':
            end = tf.shape(v)[1]
        if expected_size == 'size of v':
            return tf.reshape(tf.slice(v,(batchstart,start),(batchend-batchstart,end-start)),(batchend-batchstart,end-start))
        else:
            pre_v = tf.zeros([batchend-batchstart,start])
            post_v = tf.zeros([batchend-batchstart,expected_size-end])
            slice_v = tf.slice(v,(batchstart,start),(batchend-batchstart,end-start))
            return tf.reshape(tf.concat([pre_v,tf.cast(slice_v,tf.float32),post_v],1),(batchend-batchstart,expected_size))
    return flex_wrapper
            
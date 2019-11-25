#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorlayer.layers as l
import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import math
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from random import shuffle
import scipy.misc
import numpy as np

class CELEBA_VAE(object):
    def __init__(self,sess,epoch=30, batch_size=64, z_dim=128,
                 dataset_name='celebA', checkpoint_dir="checkpoint", 
                 result_dir="results", output_size = 64,
                 c_dim=3,beta1=0.5,test_number='vae_0808',
                 image_size=148,learning_rate=0.001,train_size=np.inf,
                 sample_step = 300,save_step = 800,is_train = False,
                 is_crop = True,load_pretrain =False):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.output_size = output_size
        self.c_dim = c_dim
        self.beta1 = beta1
        self.test_number = test_number
        self-dataset_name = dataset_name
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.train_size = train_size
        self.sample_step = sample_step
        self.save_step = save_step
        self.is_train = is_train
        self.is_crop = is_crop
        self.load_pretrain = load_pretrain
    
    def encoder(self,input_imgs, is_train = True, reuse = False):
        '''
        input_imgs: the input images to be encoded into a vector as latent representation. size here is [b_size,64,64,3]
        '''
        z_dim = self.z_dim # 512 128???
        ef_dim = 64 # encoder filter number
    
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
    
        with tf.variable_scope("encoder", reuse = reuse):
            tl.layers.set_name_reuse(reuse)
    
            net_in = l.InputLayer(input_imgs, name='en/in') # (b_size,64,64,3)
            net_h0 = l.Conv2d(net_in, ef_dim, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h0/conv2d')
            net_h0 = l.BatchNormLayer(net_h0, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='en/h0/batch_norm')
            # net_h0.outputs._shape = (b_size,32,32,64)
    
            net_h1 = l.Conv2d(net_h0, ef_dim*2, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h1/conv2d')
            net_h1 = l.BatchNormLayer(net_h1, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='en/h1/batch_norm')
            # net_h1.outputs._shape = (b_size,16,16,64*2)
    
            net_h2 = l.Conv2d(net_h1, ef_dim*4, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h2/conv2d')
            net_h2 = l.BatchNormLayer(net_h2, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='en/h2/batch_norm')
            # net_h2.outputs._shape = (b_size,8,8,64*4)
    
            net_h3 = l.Conv2d(net_h2, ef_dim*8, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h3/conv2d')
            net_h3 = l.BatchNormLayer(net_h3, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='en/h3/batch_norm')
            # net_h2.outputs._shape = (b_size,4,4,64*8)
    
            # mean of z
            net_h4 = l.FlattenLayer(net_h3, name='en/h4/flatten')
            # net_h4.outputs._shape = (b_size,8*8*64*4)
            net_out1 = l.DenseLayer(net_h4, n_units=z_dim, act=tf.identity,
                    W_init = w_init, name='en/h3/lin_sigmoid')
            net_out1 = l.BatchNormLayer(net_out1, act=tf.identity,
                    is_train=is_train, gamma_init=gamma_init, name='en/out1/batch_norm')
    
            # net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.nn.relu,
            #         W_init = w_init, name='en/h4/lin_sigmoid')
            z_mean = net_out1.outputs # (b_size,512)
    
            # log of variance of z(covariance matrix is diagonal)
            net_h5 = l.FlattenLayer(net_h3, name='en/h5/flatten')
            net_out2 = l.DenseLayer(net_h5, n_units=z_dim, act=tf.identity,
                    W_init = w_init, name='en/h4/lin_sigmoid')
            net_out2 = l.BatchNormLayer(net_out2, act=tf.nn.softplus,
                    is_train=is_train, gamma_init=gamma_init, name='en/out2/batch_norm')
            # net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.nn.relu,
            #         W_init = w_init, name='en/h5/lin_sigmoid')
            z_log_sigma_sq = net_out2.outputs + 1e-6# (b_size,512)
    
        return net_out1, net_out2, z_mean, z_log_sigma_sq
    
    def generator(self,inputs, is_train = True, reuse = False):
        '''
        generator of GAN, which can also be seen as a decoder of VAE
        inputs: latent representation from encoder. [b_size,z_dim]
        '''
        image_size = self.output_size # 64 the output size of generator
        s2, s4, s8, _ = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16) # 32,16,8,4
        gf_dim = 64
        c_dim = self.c_dim # n_color 3
        batch_size = self.batch_size # 64
    
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
    
        with tf.variable_scope("generator", reuse = reuse):
            tl.layers.set_name_reuse(reuse)
    
            net_in = l.InputLayer(inputs, name='g/in')
            net_h0 = l.DenseLayer(net_in, n_units=gf_dim*4*s8*s8, W_init=w_init,
                    act = tf.identity, name='g/h0/lin')
            # net_h0.outputs._shape = (b_size,256*8*8)
            net_h0 = l.ReshapeLayer(net_h0, shape=[-1, s8, s8, gf_dim*4], name='g/h0/reshape')
            # net_h0.outputs._shape = (b_size,8,8,256)
            net_h0 = l.BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h0/batch_norm')
    
            # upsampling
            net_h1 = l.DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s4, s4), strides=(2, 2),
                    padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
            net_h1 = l.BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h1/batch_norm')
            # net_h1.outputs._shape = (b_size,16,16,256)
    
            net_h2 = l.DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s2, s2), strides=(2, 2),
                    padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
            net_h2 = l.BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h2/batch_norm')
            # net_h2.outputs._shape = (b_size,32,32,128)
    
            net_h3 = l.DeConv2d(net_h2, gf_dim//2, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                    padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
            net_h3 = l.BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                    gamma_init=gamma_init, name='g/h3/batch_norm')
            # net_h3.outputs._shape = (b_size,64,64,32)
    
            # no BN on last deconv
            net_h4 = l.DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(1, 1),
                    padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
            # net_h4.outputs._shape = (b_size,64,64,3)
            # net_h4 = Conv2d(net_h3, c_dim, (5,5),(1,1), padding='SAME', W_init=w_init, name='g/h4/conv2d')
            logits = net_h4.outputs
            net_h4.outputs = tf.nn.tanh(net_h4.outputs)
        return net_h4, logits
    def build_model(self):
        with tf.device("/gpu:0"):
            ##========================= DEFINE MODEL ===========================##
            # the input_imgs are input for both encoder and discriminator
            input_imgs = tf.placeholder(tf.float32,[self.batch_size, self.output_size, 
                self.output_size, self.c_dim], name='real_images')
    
            # normal distribution for GAN
            self.z_p = tf.random_normal(shape=(self.batch_size, self.z_dim), mean=0.0, stddev=1.0)
            # normal distribution for reparameterization trick
            self.eps = tf.random_normal(shape=(self.batch_size, self.z_dim), mean=0.0, stddev=1.0)
            self.lr_vae = tf.placeholder(tf.float32, shape=[])
    
    
            # ----------------------encoder----------------------
            self.net_out1, self.net_out2, self.z_mean, self.z_log_sigma_sq = self.encoder(input_imgs, is_train=True, reuse=False)
    
            # ----------------------decoder----------------------
            # decode z 
            # z = z_mean + z_sigma * eps
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps)) # using reparameterization tricks
            self.gen0, _ = self.generator(self.z, is_train=True, reuse=False)
    
            # ----------------------for samples----------------------
            self.gen2, self.gen2_logits = self.generator(self.z, is_train=False, reuse=True)
            self.gen3, self.gen3_logits = self.generator(self.z_p, is_train=False, reuse=True)
                    ##========================= DEFINE TRAIN OPS =======================##
            ''''
            reconstruction loss:
            use the pixel-wise mean square error in image space
            '''
            self.SSE_loss = tf.reduce_mean(tf.square(self.gen0.outputs - input_imgs))# /self.output_size/self.output_size/3
            '''
            KL divergence:
            we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
            then compute KL divergence between z and standard normal gaussian N(0,I) 
            '''
            self.KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq),1))
    
            ### important points! ###
            # the weight between style loss(KLD) and contend loss(pixel-wise mean square error)
            VAE_loss = 0.005*self.KL_loss + self.SSE_loss # KL_loss isn't working well if the weight of SSE is too big
    
            e_vars = tl.layers.get_variables_with_name('encoder',True,True)
            g_vars = tl.layers.get_variables_with_name('generator', True, True)
            # d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
            vae_vars = e_vars+g_vars
    
            print("-------encoder-------")
            self.net_out1.print_params(False)
            print("-------generator-------")
            self.gen0.print_params(False)
    
    
            # optimizers for updating encoder, discriminator and generator
            self.vae_optim = tf.train.AdamOptimizer(self.lr_vae, beta1=self.beta1) \
                               .minimize(VAE_loss, var_list=vae_vars)
    def train_model(self):
        sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(sess)
    
        # prepare file under checkpoint_dir
        model_dir = "vae_0808"
        #  there can be many models under one checkpoine file
        save_dir = os.path.join(self.checkpoint_dir, model_dir) #'./checkpoint/vae_0808'
        tl.files.exists_or_mkdir(save_dir)
        # under current directory
        samples_1 = self.result_dir + "/" + self.test_number
        # samples_1 = self.result_dir + "/test2"
        tl.files.exists_or_mkdir(samples_1) 
    
        if self.load_pretrain == True:
            load_e_params = tl.files.load_npz(path=save_dir,name='/net_e.npz')
            tl.files.assign_params(sess, load_e_params[:24], self.net_out1)
            self.net_out1.print_params(True)
            tl.files.assign_params(sess, np.concatenate((load_e_params[:24], load_e_params[30:]), axis=0), self.net_out2)
            self.net_out2.print_params(True)
    
            load_g_params = tl.files.load_npz(path=save_dir,name='/net_g.npz')
            tl.files.assign_params(sess, load_g_params, self.gen0)
            self.gen0.print_params(True)
        
        # get the list of absolute paths of all images in dataset
        data_files = glob(os.path.join("./data", self.dataset_name, "*.jpg"))
        data_files = sorted(data_files)
        data_files = np.array(data_files) # for tl.iterate.minibatches
    
    
        ##========================= TRAIN MODELS ================================##
        iter_counter = 0
    
        # use all images in dataset in every epoch
        for epoch in range(self.epoch):
            ## shuffle data
            print("[*] Dataset shuffled!")
    
            minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=self.batch_size, shuffle=True)
            idx = 0
            batch_idxs = min(len(data_files), self.train_size) // self.batch_size
    
            while True:
                try:
                    batch_files,_ = minibatch.next()
                    batch = [self.get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
    
                    start_time = time.time()
                    vae_current_lr = self.learning_rate
    
    
                    # update
                    kl, sse, errE, _ = sess.run([self.KL_loss,self.SSE_loss,self.VAE_loss,self.vae_optim], feed_dict={self.input_imgs: batch_images, self.lr_vae:vae_current_lr})
    
    
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, vae_loss:%.8f, kl_loss:%.8f, sse_loss:%.8f" \
                            % (epoch, self.epoch, idx, batch_idxs,
                                time.time() - start_time, errE, kl, sse))
                    sys.stdout.flush()
    
                    iter_counter += 1
                    # save samples
                    if np.mod(iter_counter, self.sample_step) == 0:
                        # generate and visualize generated images
                        img1, img2 = sess.run([self.gen2.outputs, self.gen3.outputs], feed_dict={self.input_imgs: batch_images})
                        self.save_images(img1, [8, 8],
                                    './{}/train_{:02d}_{:04d}.png'.format(samples_1, epoch, idx))
    
                        # img2 = sess.run(gen3.outputs, feed_dict={input_imgs: batch_images})
                        self.save_images(img2, [8, 8],
                                    './{}/train_{:02d}_{:04d}_random.png'.format(samples_1, epoch, idx))
    
                        # save input image for comparison
                        self.save_images(batch_images,[8, 8],'./{}/input.png'.format(samples_1))
                        print("[Sample] sample generated!!!")
                        sys.stdout.flush()
    
                    # save checkpoint
                    if np.mod(iter_counter, self.save_step) == 0:
                        # save current network parameters
                        print("[*] Saving checkpoints...")
                        net_e_name = os.path.join(save_dir, 'net_e.npz')
                        net_g_name = os.path.join(save_dir, 'net_g.npz')
                        # this version is for future re-check and visualization analysis
                        net_e_iter_name = os.path.join(save_dir, 'net_e_%d.npz' % iter_counter)
                        net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)
    
    
                        # params of two branches
                        net_out_params = self.net_out1.all_params + self.net_out2.all_params
                        # remove repeat params
                        net_out_params = tl.layers.list_remove_repeat(net_out_params)
                        tl.files.save_npz(net_out_params, name=net_e_name, sess=sess)
                        tl.files.save_npz(self.gen0.all_params, name=net_g_name, sess=sess)
    
                        tl.files.save_npz(net_out_params, name=net_e_iter_name, sess=sess)
                        tl.files.save_npz(self.gen0.all_params, name=net_g_iter_name, sess=sess)
    
                        print("[*] Saving checkpoints SUCCESS!")
    
                    idx += 1
                    # print idx
                except StopIteration:
                    print 'one epoch finished'
                    break
                except Exception as e:
                    raise e
    def center_crop(self,x, crop_h, crop_w=None, resize_w=64):
        # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
        if crop_w is None:
            crop_w = crop_h # the width and height after cropped
        h, w = x.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                                   [resize_w, resize_w]) 
    
    def merge(self,images, size):
        # merge all output images(of sample size:8*8 output images of size 64*64) into one big image
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images): # idx=0,1,2,...,63
            i = idx % size[1] # column number
            j = idx // size[1] # row number
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img
    
    def transform(self,image, npx=64, is_crop=True, resize_w=64):
        if is_crop:
            cropped_image = self.center_crop(image, npx, resize_w=resize_w)
        else:
            cropped_image = image
        return np.array(cropped_image)/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN
    
    def inverse_transform(self,images):
        return (images+1.)/2. # change image pixel value(outputs from tanh in range [-1,1]) back to [0,1]
    
    def imread(self,path, is_grayscale = False):
        if (is_grayscale):
            return scipy.misc.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
        else:
            return scipy.misc.imread(path).astype(np.float) # [width,height,color_dim]
    
    def imsave(self,images, size, path):
        return scipy.misc.imsave(path, self.merge(images, size))
    
    def get_image(self,image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
        return self.transform(self.imread(image_path, is_grayscale), image_size, is_crop, resize_w)
    
    def save_images(self,images, size, image_path):
        # size indicates how to arrange the images to form a big summary image
        # images: [batchsize,height,width,color]
        # example: save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(self.result_dir, epoch, idx))
        return self.imsave(self.inverse_transform(images), size, image_path)
    
    def save_images_256(self,images, size, image_path):
        images = self.inverse_transform(images)
        h, w = 64, 64 # 256,256
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images): # idx=0,1,2,...,63
            image = scipy.misc.imresize(image,[h,w])
            i = idx % size[1] # column number
            j = idx // size[1] # row number
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return scipy.misc.imsave(image_path, img)


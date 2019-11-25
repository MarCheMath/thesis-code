#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:20:52 2019

@author: marche
"""
import numpy as np
import tensorflow as tf
from generative_model_collections.VAE import VAE as VAE

path = './estimated/fashion-mnist/full-input/project/0.0/784/VAEflex250/0.0_1.0_0.0_adam_0.1_0.9_False_10000_10_0.01non-squared_1_previous-and-random_1_1/'
directory = './generative_model_collections/checkpoint/'
k=250

z_val = np.load(path+'z_rec.pkl',allow_pickle=True)
print z_val.shape


sess = tf.Session()
vae=VAE(sess,25,64,k,'fashion-mnist',directory,'','')
vae.z= tf.Variable(tf.cast(z_val,tf.float32), name='z')
x = vae.decoder(vae.z,is_training=False,reuse=False)
vae.load(directory)
sess.run(x,feed_dict={vae.z:z_val})
print(x)
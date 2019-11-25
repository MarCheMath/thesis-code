#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:20:52 2019

@author: marche
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import copy
#from generative_model_collections.VAE import VAE as VAE
import generative_model_collections.flex_wrapper_DIEHARD as wrapper

def execute():
    K=250
    grid=[5,10,20,40,60,100,200,250]
    grid_str = str(grid)[1:-1].replace(', ','-')
    K_str = str(K)+'_'
    vae=wrapper.init('VAE',training=False)(sess='only mode def',epoch='only mode def',batch_size=64,dataset_name='fashion-mnist',checkpoint_dir='./generative_model_collections/checkpoint',result_dir='./generative_model_collections/results',log_dir='./generative_model_collections/logs',grid=grid,repetition_bol='False')
    path = './estimated/fashion-mnist/full-input/project/0.0/784/VAEflex{}/0.0_1.0_0.0_adam_0.1_0.9_False_1000_10_0.01non-squared_1_previous-and-random_{}1_1/'.format(grid_str,K_str)
    directory = './generative_model_collections/checkpoint/'
    
    print path+'z_rec.pkl'
    z_val = np.load(path+'z_rec.pkl',allow_pickle=True)
    z_val=vae.split_and_merge(z_val)
    
    
    sess = tf.Session()
    z_val = sess.run(z_val)
    print z_val.shape
    vae.z= tf.Variable(tf.cast(z_val,tf.float32), name='z')
    x = vae.decoder(vae.z,is_training=False,reuse=False)
    vae.saver = tf.train.Saver()
    var_list = {var.op.name:var for var in tf.global_variables() if 'decoder' in var.op.name or 'de_' in var.op.name}
    
    vae.load(directory,sess,var_list,'obey_grid')
    counter=0
    for k in [5,10,20,40,60,100,200,250]:
        z_val_k=copy.deepcopy(z_val)
        z_val_k[:,k:]=0
        x_val=sess.run(x,feed_dict={vae.z:z_val_k})
        ind = np.asarray(range(len(x_val)))%8==7
        images = x_val[ind]
        big_image = np.zeros((28*8,28*8))
        for i in range(8):
            for j in range(8):
                big_image[28*i:28*(i+1),28*j:28*(j+1)] = np.squeeze(images[8*i+j])
        save_plot(big_image,'./results/vaeflex_fashion_{}.png'.format(counter))
#        fig =plt.figure()
#        fig.patch.set_visible(False)
#    #    plt.imshow(x_val[15].reshape((28,28)),cmap='gray')
#    #    plt.imshow(images[1].reshape((28,28)),cmap='gray')
#        plt.imshow(big_image,cmap='gray')
#        #plt.show()
#        plt.axis('off')
#        #with open('./results/vaeflex_fashion_{}.png'.format(counter), 'w') as outfile:
#        #    fig.canvas.print_png(outfile)
#    #    plt.savefig('./results/vaeflex_fashion_{}.png'.format(counter), bbox_inches='tight')
#        plt.savefig('./results/vaeflex_fashion_{}.png'.format(counter))
        counter=counter+1



def save_plot(data,filename):
    fig =plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap='gray')
    plt.savefig(filename, dpi = 28*8) 
    plt.close()


execute()
##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Tue Oct 22 15:20:52 2019
#
#@author: marche
#"""
#import numpy as np
#import tensorflow as tf
#from generative_model_collections.VAE import VAE as VAE
#
#path = './estimated/fashion-mnist/full-input/project/0.0/784/VAEflex784/0.0_1.0_0.0_adam_0.1_0.9_False_10000_10_0.01non-squared_1_previous-and-random_1_1_784/'
#directory = './generative_model_collections/checkpoint/'
#k=784
#
#z_val = np.load(path+'z_rec.pkl',allow_pickle=True)
#print z_val.shape
#
#
#sess = tf.Session()
#vae=VAE(sess,25,64,k,'fashion-mnist',directory,'','')
#vae.z= tf.Variable(tf.cast(z_val,tf.float32), name='z')
#x = vae.decoder(vae.z,is_training=False,reuse=False)
#vae.load(directory)
#sess.run(x,feed_dict={vae.z:z_val})
#print(x)
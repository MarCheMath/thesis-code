"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103, R0914, C0111

import os
import sys
import tensorflow as tf
import numpy as np
import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mnist_vae.src import model_def as mnist_vae_model_def
from mnist_e2e.model_def import end_to_end


def construct_gen(hparams, model_def):

    model_hparams = model_def.Hparams()
    model_hparams.grid = hparams.grid
    model_hparams.stdv = hparams.stdv
    model_hparams.mean = hparams.mean
    
    z = model_def.get_z_var(model_hparams, hparams.batch_size)
    _, x_hat,b3 = model_def.generator(model_hparams, z, 'gen', reuse=hparams.bol)

    restore_vars = model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return z, x_hat, restore_path, restore_dict,b3

def construct_en(hparams, model_def,x):

    model_hparams = model_def.Hparams()
    model_hparams.n_z = hparams.n_z

    z_mean,z_log_sigma_sq = model_def.encoder(model_hparams, x, 'enc', reuse=False)
    z_std = tf.sqrt(tf.exp(z_log_sigma_sq))

    restore_vars = model_def.enc_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    
    return z_std, z_mean, restore_path, restore_dict

def validate(z,x,hparams):    
#    z = np.load(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'z.npy')
    model_hparams = mnist_vae_model_def.Hparams()
    model_hparams.stdv = hparams.stdv
    model_hparams.mean = hparams.mean
    
    _, x_hat,b3 = mnist_vae_model_def.generator(model_hparams, z, 'gen', reuse=hparams.bol)
    
    restore_vars = mnist_vae_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
    
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)
    x_hat_val = sess.run(x_hat)
    sess.close()
    tf.reset_default_graph()
    print(np.linalg.norm(x_hat_val-x))


def applicate_encoder(hparams,x_batch):
    """Sample random images from the generator"""

    # encode
    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')

    # sample
    eps = tf.random_normal((hparams.batch_size, hparams.n_z), 0, 1, dtype=tf.float32)
    z_std, z_mean, restore_path, restore_dict = construct_en(hparams,mnist_vae_model_def,x_ph)
    z = z_mean + z_std * eps
    
    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)
    feed_dict = {x_ph: x_batch}
    z_r = sess.run( z, feed_dict=feed_dict)
#    print(np.mean(sess.run( z_mean, feed_dict=feed_dict)))
#    print(np.mean(sess.run( z_std, feed_dict=feed_dict)))
    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()

    return z_r

def applicate_decoder(hparams,z_batch):
    """Sample random images from the generator"""

    # encode
    z = tf.placeholder(tf.float32, [None, hparams.n_z], name='z')
    print(z.shape)
    model_hparams = mnist_vae_model_def.Hparams()
    model_hparams.n_z = hparams.n_z
    _,x,_ = mnist_vae_model_def.generator(model_hparams, z, 'gen', False)

    restore_vars = mnist_vae_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
    
    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)
    feed_dict = {z: z_batch}
    x_r = sess.run(x, feed_dict=feed_dict)
#    print(np.mean(sess.run( z_mean, feed_dict=feed_dict)))
#    print(np.mean(sess.run( z_std, feed_dict=feed_dict)))
    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()

    return x_r

def vae_gen(hparams):
    return construct_gen(hparams, mnist_vae_model_def)

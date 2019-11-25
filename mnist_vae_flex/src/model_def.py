# This file based on : https://jmetzen.github.io/notebooks/vae.ipynb
# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902


import tensorflow as tf
import utils
import numpy as np


class Hparams(object):
    def __init__(self):
        self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
        self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
        self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
        self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
        self.n_input = 784           # MNIST data input (img shape: 28*28)
#        self.n_z = 20
        self.n_z = 0
#        self.grid = np.concatenate(([5,10], range(20,300,self.n_z)),axis=None)             # dimensions of latent spaces
        self.transfer_fct = tf.nn.softplus
        #self.stdv = 10
        #self.mean = 0


def _encoder_(hparams, x_ph, scope_name, reuse):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', initializer=utils.xavier_init(hparams.n_input, hparams.n_hidden_recog_1))
        b1 = tf.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_recog_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(x_ph, w1) + b1)

        w2 = tf.get_variable('w2', initializer=utils.xavier_init(hparams.n_hidden_recog_1, hparams.n_hidden_recog_2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_recog_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)
        
        W3 = tf.get_variable('W3', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.grid[-1]))
        B3 = tf.get_variable('B3', initializer=tf.zeros([hparams.grid[-1]], dtype=tf.float32))
        z_mean = tf.matmul(hidden2, W3) + B3

        W4 = tf.get_variable('W4', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.grid[-1]))
        B4 = tf.get_variable('B4', initializer=tf.zeros([hparams.grid[-1]], dtype=tf.float32))
        z_log_sigma_sq = tf.matmul(hidden2, W4) + B4

    return z_mean, z_log_sigma_sq

def encoder_i(hparams, x_ph, scope_name, reuse,relative=True):
    z_mean, z_log_sigma_sq = _encoder_(hparams, x_ph, scope_name, reuse)
    eps = tf.random_normal((hparams.batch_size, hparams.grid[-1]), 0, 1, dtype=tf.float32)
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    z = z_mean + z_sigma * eps
    
    z_comp_list = [[],[]]
    z_list = []
    for i in range(len(hparams.grid)):
        z_comp_list[0].append((slicer_enc(hparams,i,None,z_mean)))
        z_comp_list[1].append(slicer_enc(hparams,i,None,z_log_sigma_sq))
        z_list.append(slicer_enc(hparams,i,None,z,relative))
    return z_list, z_comp_list

def generator_i(hparams, z, scope_name, reuse,i,relative=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        
        try:
            W1 = tf.get_variable('W1', initializer=utils.xavier_init(hparams.grid[-1], hparams.n_hidden_gener_1))
        except ValueError:
            scope.reuse_variables()
            W1 = tf.get_variable('W1', initializer=utils.xavier_init(hparams.grid[-1], hparams.n_hidden_gener_1))        
        w1 = slicer_dec(hparams,i,W1,None,relative)
#        hparams.track.append(w1)
        b1 = tf.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_gener_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(z, w1) + b1)

        w2 = tf.get_variable('w2', initializer=utils.xavier_init(hparams.n_hidden_gener_1, hparams.n_hidden_gener_2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_gener_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable('w3', initializer=utils.xavier_init(hparams.n_hidden_gener_2, hparams.n_input))
        b3 = tf.get_variable('b3', initializer=tf.zeros([hparams.n_input], dtype=tf.float32))
        
        logits = tf.matmul(hidden2, w3) + b3
        x_reconstr_mean = tf.nn.sigmoid(logits)

    return logits, x_reconstr_mean, b3

def model(hparams,x_ph,scopeNames,reuses):
    z_list, z_comp_list = encoder_i(hparams, x_ph, scopeNames[0], reuses[0])
    
#    W1 = tf.get_variable('W1', initializer=utils.xavier_init(hparams.grid[-1], hparams.n_hidden_gener_1))
    
    logits_list = []
    x_reconstr_mean_list = []
    b_list = []
    loss_list = []
#    hparams.track=[]
#    hparams.x_ph_ref=[]
#    hparams.logits_ref=[]
#    reuses[1]=False
    for i in range(len(hparams.grid)):
#        output=generator_i(hparams, z_list[i], scopeNames[1], reuses[1],slicer_dec(hparams,i,W1,None))
        output=generator_i(hparams, z_list[i], scopeNames[1], reuses[1],i)
#        reuses[1]=True
        logits_list.append(output[0])
        x_reconstr_mean_list.append(output[1])
        b_list.append(output[2])
#        hparams.x_ph_ref.append(x_ph)
#        hparams.logits_ref.append(output[0])
        loss_list.append(__get_loss__(x_ph, output[0], z_comp_list[0][i], z_comp_list[1][i]))
    hparams.zl=z_list
    return logits_list, x_reconstr_mean_list, b_list, loss_list
        

#def slicer_enc_i(hparams,i,W,B):
#    slice_list = [0, hparams.grid]
#    return (tf.slice(W,(slice_list[i],0),(slice_list[i+1],tf.shape(W)[1])), \
#            tf.slice(B,(slice_list[i],),(slice_list[i+1],)))
#
#def slicer_dec_i(hparams,i,W,B):
#    slice_list = [0, hparams.grid]
#    return (tf.slice(W,(0,slice_list[i]),(tf.shape(W)[1],slice_list[i+1])), \
#            tf.slice(B,(slice_list[i],),(slice_list[i+1],)))
    
def slicer_enc(hparams,i,W,B,relative=True):
    slice_list = np.concatenate(([0], hparams.grid),axis=None)
    if W == None:
        if relative:
            return tf.slice(B,(0,0),(tf.shape(B)[0],slice_list[i+1]))
        else:
            return tf.slice(B,(0,0),(tf.shape(B)[0],i))
    elif B == None:
        if relative:
            return tf.slice(W,(0,0),(tf.shape(W)[1],slice_list[i+1])) #always kinda transposed used
        else:
            return tf.slice(W,(0,0),(tf.shape(W)[1],i))
    else:
        if relative:
            return (tf.slice(W,(0,0),(tf.shape(W)[1],slice_list[i+1])), \
                    tf.slice(B,(0,0),(tf.shape(B)[0],slice_list[i+1])))
        else:
            return (tf.slice(W,(0,0),(tf.shape(W)[1],i)), \
                    tf.slice(B,(0,0),(tf.shape(B)[0],i)))

def slicer_dec(hparams,i,W,B,relative=True):
    slice_list = np.concatenate(([0], hparams.grid),axis=None)
    if W == None:
        if relative:
            return tf.slice(B,(0,0),(tf.shape(B)[0],slice_list[i+1]))
        else:
            return tf.slice(B,(0,0),(tf.shape(B)[0],i))
    elif B == None:
        if relative:
            return tf.slice(W,(0,0),(slice_list[i+1],tf.shape(W)[1]))
        else:
            return tf.slice(W,(0,0),(i,tf.shape(W)[1]))
    else:
        if relative:
            return (tf.slice(W,(0,0),(slice_list[i+1],tf.shape(W)[1])), \
                    tf.slice(B,(0,0),(tf.shape(B)[0],slice_list[i+1])))
        else:
            return (tf.slice(W,(0,0),(i,tf.shape(W)[1])), \
                    tf.slice(B,(0,0),(tf.shape(B)[0],i)))
            
def __get_loss__(x, logits, z_mean, z_log_sigma_sq):    
    reconstr_losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), 1)
    latent_losses = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    total_loss = tf.reduce_mean(reconstr_losses + latent_losses)
    return total_loss
    

def get_z_var(hparams, batch_size, thick = 'notdefined', stdv = 'notdefined', mean = 'notdefined'):
    if thick == 'notdefined':
        thick = hparams.grid[-1]
    if stdv == 'notdefined':
        stdv = hparams.stdv
    if mean == 'notdefined':
        mean = hparams.mean
    thick = int(thick)
    z = tf.Variable(tf.random_normal((batch_size, thick),stddev=stdv,mean=mean), name='z')
    return z

def applicate_encoder(hparams,x_batch,pos_in_train_grid):
    """Sample random images from the generator"""

    # encode
    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')

    # sample
    z_list, _  = encoder_i(hparams,x_ph,'enc',False)
    z = z_list[pos_in_train_grid]
    restore_vars = enc_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    
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


def gen_restore_vars():
    restore_vars = ['gen/W1',
                    'gen/b1',
                    'gen/w2',
                    'gen/b2',
                    'gen/w3',
                    'gen/b3']
    return restore_vars

def enc_restore_vars():
    restore_vars = ['enc/w1',
                    'enc/b1',
                    'enc/w2',
                    'enc/b2',
                    'enc/W3',
                    'enc/B3',
                    'enc/W4',
                    'enc/B4']
    return restore_vars
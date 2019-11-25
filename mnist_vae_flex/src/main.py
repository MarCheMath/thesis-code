# This file based on : https://jmetzen.github.io/notebooks/vae.ipynb
# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902

import os
import numpy as np
import tensorflow as tf
import utils
import model_def
import data_input
from argparse import ArgumentParser


def main(hparams):
#    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Set up some stuff according to hparams
    utils.set_up_dir(hparams.ckpt_dir)
    utils.set_up_dir(hparams.sample_dir)
    utils.print_hparams(hparams)
    # encode
    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')
    
    _,x_reconstr_mean,_, loss_list = model_def.model(hparams,x_ph,['enc','gen'],[False,False])
    total_loss = tf.add_n(loss_list, name='total_loss')
    
    
#    z_mean, z_log_sigma_sq = model_def.encoder(hparams, x_ph, 'enc', reuse=False)
#
#    # sample
#    eps = tf.random_normal((hparams.batch_size, hparams.n_z), 0, 1, dtype=tf.float32)
#    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
#    z = z_mean + z_sigma * eps
#
#    # reconstruct
#    logits, x_reconstr_mean, _ = model_def.generator(hparams, z, 'gen', reuse=False)
#
    # generator sampler
    z_ph = tf.placeholder(tf.float32, [None, hparams.grid[-1]], name='x_ph')
    x_sample = []
    for i in range(len(hparams.grid)):
        _, x_sample_tmp, _ = model_def.generator_i(hparams, model_def.slicer_dec(hparams,i,None,z_ph), 'gen', True,i) 
        x_sample.append(x_sample_tmp)
#    _, x_sample, _ = model_def.generator(hparams, z_ph, 'gen', reuse=True)

#    # define loss and update op
#    total_loss = model_def.get_loss(x_ph, logits, z_mean, z_log_sigma_sq)
    opt = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    update_op = opt.minimize(total_loss)
#    print([el.name for el in tf.trainable_variables()])

    # Sanity checks
    for var in tf.global_variables():
        print(var.op.name)
    print('')
#    print([o.name for o in tf.trainable_variables()])

    # Get a new session
    sess = tf.Session()

    # Model checkpointing setup
    model_saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    start_epoch = utils.try_restore(hparams, sess, model_saver)

    # Get data iterator
    iterator = data_input.mnist_data_iteratior(dataset=hparams.dataset)

    # Training
    for epoch in range(start_epoch+1, hparams.training_epochs):
        avg_loss = 0.0
        num_batches = hparams.num_samples // hparams.batch_size
        batch_num = 0
        for (x_batch_val, _) in iterator(hparams, num_batches):
            batch_num += 1
            feed_dict = {x_ph: x_batch_val}
            _, loss_val = sess.run([update_op, total_loss], feed_dict=feed_dict)
            avg_loss += loss_val / hparams.num_samples * hparams.batch_size

            if batch_num % 100 == 0:
                x_reconstr_mean_val = sess.run(x_reconstr_mean, feed_dict={x_ph: x_batch_val})

                z_val = np.random.randn(hparams.batch_size, hparams.grid[-1])
                x_sample_val = sess.run(x_sample, feed_dict={z_ph: z_val})
#                print(sess.run(hparams.track[0], feed_dict={z_ph: z_val}))
#                print(sess.run(hparams.track[1], feed_dict={z_ph: z_val}))
#                s1 = sess.run(hparams.x_ph_ref[0], feed_dict={x_ph: x_batch_val})
#                s2 = sess.run(hparams.logits_ref[0], feed_dict={x_ph: x_batch_val})
#                s3 = sess.run(loss_list, feed_dict={x_ph: x_batch_val})
#                print(s1.shape)
#                print(s2.shape)
#                print(s3)
                utils.save_images(np.reshape(x_batch_val, [-1, 28, 28]), \
                                      [10, 10], \
                                      '{}/orig_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                for i in range(len(hparams.grid)):
                    utils.save_images(np.reshape(x_reconstr_mean_val[i], [-1, 28, 28]),
                                      [10, 10],
                                      '{}/reconstr_{}_{:02d}_{:04d}.png'.format(hparams.sample_dir, hparams.grid[i], epoch, batch_num))
                    
                    utils.save_images(np.reshape(x_sample_val[i], [-1, 28, 28]),
                                      [10, 10],
                                      '{}/sampled_{}_{:02d}_{:04d}.png'.format(hparams.sample_dir, hparams.grid[i], epoch, batch_num))


        if epoch % hparams.summary_epoch == 0:
            print("Epoch:", '%04d' % (epoch), 'Avg loss = {:.9f}'.format(avg_loss))

        if epoch % hparams.ckpt_epoch == 0:
            save_path = os.path.join(hparams.ckpt_dir, '{}_vae_model_flex_hid'.format(hparams.dataset)+str('_'.join(map(str,hparams.grid))))
            model_saver.save(sess, save_path, global_step=epoch)

    save_path = os.path.join(hparams.ckpt_dir, '{}_vae_model_flex_hid'.format(hparams.dataset)+str('_'.join(map(str,hparams.grid))))
    model_saver.save(sess, save_path, global_step=hparams.training_epochs-1)


if __name__ == '__main__':

    PARSER = ArgumentParser()

    # Pretrained model
#    PARSER.add_argument('--n_z', type=int, default=20, help='Hidden dimension n_z of the model')
    PARSER.add_argument('--n_z', type=int, default=0, help='NOT USED HERE')
#    PARSER.add_argument('--grid', type=int, default=np.concatenate(([5,10], range(20,300,PARSER.parse_args().n_z)),axis=None), help='Hidden dimension n_z of the model')
    PARSER.add_argument('--grid', type=str, default="5 10 15 20 40 60 80 100", help='Hidden dimension n_z of the model')
    PARSER.add_argument('--training-epochs', type=int, default=100, help='Number of training epochs')
    PARSER.add_argument('--dataset', type=str, default='mnist', help='Used dataset to train on (28x28) required')

    HPARAMS = model_def.Hparams()
    HPARAMS.__dict__.update(PARSER.parse_args().__dict__.copy())
#    HPARAMS.n_z = PARSER.parse_args().n_z
    HPARAMS.grid = PARSER.parse_args().grid
    HPARAMS.grid = map(int,HPARAMS.grid.split())
    

    HPARAMS.num_samples = 60000
    HPARAMS.learning_rate = 0.001
    HPARAMS.batch_size = 100
#    HPARAMS.training_epochs = 100
    HPARAMS.summary_epoch = 1
    HPARAMS.ckpt_epoch = 5


    HPARAMS.ckpt_dir = './{}_vae_flex/models/{}-vae-flex-{}/'.format(HPARAMS.dataset,HPARAMS.dataset,'-'.join(map(str,HPARAMS.grid)))
    HPARAMS.sample_dir = './{}_vae_flex/samples/{}-vae-flex-{}/'.format(HPARAMS.dataset,HPARAMS.dataset,'-'.join(map(str,HPARAMS.grid)))

    main(HPARAMS)

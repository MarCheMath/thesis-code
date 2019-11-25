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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Set up some stuff according to hparams
    utils.set_up_dir(hparams.ckpt_dir)
    utils.set_up_dir(hparams.sample_dir)
    utils.print_hparams(hparams)

    # encode
    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')
    z_mean, z_log_sigma_sq = model_def.encoder(hparams, x_ph, 'enc', reuse=False)

    # sample
    eps = tf.random_normal((hparams.batch_size, hparams.n_z), 0, 1, dtype=tf.float32)
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    z = z_mean + z_sigma * eps

    # reconstruct
    logits, x_reconstr_mean, _ = model_def.generator(hparams, z, 'gen', reuse=False)

    # generator sampler
    z_ph = tf.placeholder(tf.float32, [None, hparams.n_z], name='x_ph')
    _, x_sample, _ = model_def.generator(hparams, z_ph, 'gen', reuse=True)

    # define loss and update op
    total_loss = model_def.get_loss(hparams,x_ph, logits, z_mean, z_log_sigma_sq)
    opt = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    update_op = opt.minimize(total_loss)

    # Sanity checks
    for var in tf.global_variables():
        print(var.op.name)
    print('')

    # Get a new session
    sess = tf.Session()

    # Model checkpointing setup
    model_saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    start_epoch = utils.try_restore(hparams, sess, model_saver)

    # Get data iterator
    iterator = data_input.mnist_data_iteratior()

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

                z_val = np.random.randn(hparams.batch_size, hparams.n_z)
                x_sample_val = sess.run(x_sample, feed_dict={z_ph: z_val})

                utils.save_images(np.reshape(x_reconstr_mean_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/reconstr_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                utils.save_images(np.reshape(x_batch_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/orig_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                utils.save_images(np.reshape(x_sample_val, [-1, 28, 28]),
                                  [10, 10],
                                  '{}/sampled_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))


        if epoch % hparams.summary_epoch == 0:
            print("Epoch:", '%04d' % (epoch), 'Avg loss = {:.9f}'.format(avg_loss))

        if epoch % hparams.ckpt_epoch == 0:
            save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model_hid'+str(hparams.n_z))
            model_saver.save(sess, save_path, global_step=epoch)

    save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model'+str(hparams.n_z))
    model_saver.save(sess, save_path, global_step=hparams.training_epochs-1)


if __name__ == '__main__':

    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--n_z', type=int, default=20, help='Hidden dimension n_z of the model')
    PARSER.add_argument('--gauss_sigma_sq', type=float, default=1., help='Hidden dimension n_z of the model')
    PARSER.add_argument('--weight_of_internal_distribution', type=float, default=1., help='Hidden dimension n_z of the model')

    HPARAMS = model_def.Hparams()
    HPARAMS.n_z = PARSER.parse_args().n_z
    HPARAMS.gauss_sigma_sq = PARSER.parse_args().gauss_sigma_sq
    HPARAMS.weight_of_internal_distribution = PARSER.parse_args().weight_of_internal_distribution

    HPARAMS.num_samples = 60000
    HPARAMS.learning_rate = 0.001
    HPARAMS.batch_size = 100
    HPARAMS.training_epochs = 100
    HPARAMS.summary_epoch = 1
    HPARAMS.ckpt_epoch = 5

    HPARAMS.ckpt_dir = './models/mnist-vae{}-{}/'.format(HPARAMS.n_z,HPARAMS.weight_of_internal_distribution)
    HPARAMS.sample_dir = './samples/mnist-vae{}-{}/'.format(HPARAMS.n_z,HPARAMS.weight_of_internal_distribution)

    main(HPARAMS)
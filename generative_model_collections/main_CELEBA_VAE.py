#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from CELEBA_VAE import CELEBA_VAE


pp = pprint.PrettyPrinter()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
Tensorlayer implementation of VAE
'''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [5]") 
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [108]")
# flags.DEFINE_integer("decoder_output_size", 64, "The size of the output images to produce from decoder[64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 128, "Dimension of latent representation vector from. [2048]")
flags.DEFINE_integer("sample_step", 300, "The interval of generating sample. [300]")
flags.DEFINE_integer("save_step", 800, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset_name", "celebA", "The name of dataset [celebA]")
flags.DEFINE_string("test_number", "vae_0808", "The number of experiment [test2]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("result_dir", "results", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
# flags.DEFINE_integer("class_dim", 4, "class number for auxiliary classifier [5]") 
#flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("load_pretrain",False, "Default to False;If start training on a pretrained net, choose True")
FLAGS = flags.FLAGS

def main():
    pp.pprint(FLAGS.__flags)

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.result_dir)
    model = CELEBA_VAE(epoch=FLAGS.epoch,
                       batch_size=FLAGS.batch_size, 
                       z_dim=FLAGS.z_dim,
                       dataset_name=FLAGS.dataset_name,
                       checkpoint_dir=FLAGS.checkpoint_dir,
                       result_dir=FLAGS.result_dir,
                       output_size=FLAGS.output_size,
                       c_dim=FLAGS.c_dim,
                       beta1=FLAGS.beta1,
                       test_number=FLAGS.test_number, 
                       image_size=FLAGS.image_size,
                       learning_rate=FLAGS.learning_rate,
                       train_size=FLAGS.train_size,
                       sample_step = FLAGS.sample_step,
                       save_step = FLAGS.save_step,
                       is_train = FLAGS.is_train,
                       is_crop = FLAGS.is_crop,
                       load_pretrain = FLAGS.load_pretrain
                       )
    model.build_model()
    training_start_time = time.time()
    model.train_model()
    training_end_time = time.time()
    print("The processing time of program is : {:.2f}mins".format((training_end_time-training_start_time)/60.0))

if __name__ == '__main__':
    tf.app.run()
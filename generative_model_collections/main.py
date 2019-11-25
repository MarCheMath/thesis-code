import os

## GAN Variants
from GAN import GAN
from CGAN import CGAN
from infoGAN import infoGAN
from ACGAN import ACGAN
from EBGAN import EBGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from DRAGAN import DRAGAN
from LSGAN import LSGAN
from BEGAN import BEGAN

## VAE Variants
from VAE import VAE
from CVAE import CVAE

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan-type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z-dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--grid', type=int, nargs='+', default=-1,
                        help='For flexible version trained steps')
    parser.add_argument('--repetition-bol', type=str, default = 'False', 
                        help='Whether to repeat generator training as many times as grid points')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    
    for field in args.__dict__:
        print('{} : {}'.format(field, getattr(args,field)))
    # open session
    models = [GAN, CGAN, infoGAN, ACGAN, EBGAN, WGAN, WGAN_GP, DRAGAN,
              LSGAN, BEGAN, VAE, CVAE]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                if args.grid != [-1] and args.grid != -1:
                    import flex_wrapper_DIEHARD
                    model = flex_wrapper_DIEHARD.init(args.gan_type)
                    gan = model(sess,
                                epoch=args.epoch,
                                batch_size=args.batch_size,
                                dataset_name=args.dataset,
                                checkpoint_dir=args.checkpoint_dir,
                                result_dir=args.result_dir,
                                log_dir=args.log_dir,
                                grid = args.grid,
                                repetition_bol = args.repetition_bol)                    
                else:
                    gan = model(sess,
                                epoch=args.epoch,
                                batch_size=args.batch_size,
                                z_dim=args.z_dim,
                                dataset_name=args.dataset,
                                checkpoint_dir=args.checkpoint_dir,
                                result_dir=args.result_dir,
                                log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()

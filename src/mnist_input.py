"""Inputs for MNIST dataset"""

import math
import numpy as np
import mnist_model_def
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
import os, gzip

NUM_TEST_IMAGES = 10000


def get_random_test_subset(mnist, sample_size,seed='no_seed'):
    """Get a small random subset of test images"""
    if seed != 'no_seed':
        st0 = np.random.get_state()
        np.random.seed(int(seed))
    idxs = np.random.choice(NUM_TEST_IMAGES, sample_size)
    images = [mnist.test.images[idx] for idx in idxs]
    labels = [mnist.test.labels[idx] for idx in idxs]
    images = {i: image for (i, image) in enumerate(images)}
    labels = {i: labels for (i, labels) in enumerate(labels)}
    if seed != 'no_seed':
        np.random.set_state(st0)
    return (images,labels)


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    # Create the generator
    z, x_hat, restore_path, restore_dict,b3 = mnist_model_def.vae_gen(hparams)

    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    images = {}
    zs=[]
    counter = 0
    rounds = int(math.ceil(hparams.num_input_images/hparams.batch_size))
    for _ in range(rounds):
        z, images_mat = sess.run([z, x_hat])
        #print(sess.run(b3))
        for (_, image) in enumerate(images_mat):
            if counter < hparams.num_input_images:
                images[counter] = image
                zs.append(z[counter])
                counter += 1

    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()
    hparams.z_from_gen=np.asarray(zs)
    hparams.images_mat = images_mat
    np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'z.npy',hparams.z_from_gen)
    np.save(utils.get_checkpoint_dir(hparams, hparams.model_types[0])+'images.npy',hparams.images_mat)

    return images


def model_input(hparams):
    """Create input tensors"""
    if hparams.dataset == 'mnist':
        mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    elif hparams.dataset == 'fashion-mnist':
        mnist = load_mnist('fashion-mnist')
    else:
        raise NotImplementedError('only mnist and fashion-mnist are allowed choices!')

    if hparams.input_type == 'full-input':
        images = {i: image for (i, image) in enumerate(mnist.test.images[:hparams.num_input_images])}
        labels = {i: label for (i, label) in enumerate(mnist.test.labels[:hparams.num_input_images])}
        #if hparams.dict_flag == True:
            #images = {key:hparams.key_field[key] for key in hparams.key_field.keys()[:hparams.num_input_images]}
            #images = {i: image for (i, image) in enumerate(mnist.test.images[:hparams.num_input_images])}
    elif hparams.input_type == 'dict-input':
#        idx = np.asarray(hparams.key_field.keys())
#        images = {idx[i]: image for (i, image) in enumerate(mnist.test.images[idx])}
#        print(hparams.key_field.keys()[:hparams.num_input_images])
#        print(type(hparams.key_field.keys()))
#        a=hparams.key_field.keys()[:hparams.num_input_images]
#        images = hparams.key_field[a]
        images = {key:hparams.key_field[key] for key in hparams.key_field.keys()[:hparams.num_input_images]}
        try:
            labels = {key:hparams.label_field[key] for key in hparams.label_field.keys()[:hparams.num_input_images]}
        except:
            labels = None
        #images = {i: image for (i, image) in enumerate(mnist.test.images[:hparams.num_input_images])}
    elif hparams.input_type == 'random-test':
        images, labels = get_random_test_subset(mnist, hparams.num_input_images,seed=hparams.input_seed)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams) 
        labels = None
#        mnist_model_def.validate(hparams.z_from_gen,images,hparams)
    else:
        raise NotImplementedError
    
    return (images,labels)


def load_mnist(dataset_name):
    import os,gzip
    import numpy as np
    class data_container(object):
        class container_field(object):
            def next_batch(self,batch):
                ret_valx = self.images[self.pos:self.pos+batch]
                ret_valy = self.labels[self.pos:self.pos+batch]
                self.pos = self.pos + batch
                return (ret_valx,ret_valy)
            def __init__(self):
                self.pos=0
        def __init__(self,training_data,training_label,test_data,test_label):
            test = data_container.container_field()
            test.images = test_data
            test.labels = test_label
            training = data_container.container_field()
            training.images = training_data
            training.labels = training_label
            self.test = test
            self.train = training
            
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = np.float32(np.asarray(data.reshape((60000, -1)))/255)

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000)).astype(int)

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = np.float32(np.asarray(data.reshape((10000, -1)))/255)

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000)).astype(int)

    y_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec[i, teY[i]] = 1.0
    teY = y_vec
        
    y_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec[i, trY[i]] = 1.0    
    trY = y_vec
    return data_container(trX,trY,teX,teY)
# Get input



def mnist_data_iteratior(dataset='mnist'):
    if dataset == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        #mnist = load_mnist('mnist')
        mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    elif dataset == 'fashion-mnist':
        import fashion_mnist_loader
        mnist = fashion_mnist_loader.read_data_sets('../data/fashion-mnist', one_hot=True)
        #mnist = load_mnist('fashion-mnist')
    else:
        raise NotImplementedError("The dataset name {} does not exist, only 'mnist' and 'mnistfashion' are allowed".format(dataset))
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.train.next_batch(hparams.batch_size)
    return iterator

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
            
    data_dir = os.path.join("../data", dataset_name)

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
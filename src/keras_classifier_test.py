import keras
def test(filepath):	
	import numpy as np
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("data/MNIST", one_hot=True)
	x_rec = np.reshape(np.load(filepath,allow_pickle=True),(-1,28,28,1))
	labels = mnist.test.labels
	labels = labels[:64,:]
	model = keras.models.load_model('final_model.h5')
	_, acc = model.evaluate(x_rec, labels, verbose=0)
	print('> %.3f' % (acc * 100.0))

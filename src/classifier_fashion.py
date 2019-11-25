#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#authoer: Christopher Masch
import numpy as np
import matplotlib.pyplot as plt
import codecs, cv2, datetime, glob, itertools, keras, os, pickle
import re, sklearn, string, sys, tensorflow, time
from random import randint
from keras import backend as K, regularizers, optimizers
from keras.models import load_model, Sequential
from keras.layers import MaxPooling2D, Convolution2D, Activation, Dropout, Flatten, Dense, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

K.set_image_dim_ordering('tf')

#######################
# Dimension of images #
#######################
img_width  = 28
img_height = 28
channels   = 1

######################
# Parms for learning #
######################
batch_size = 250
num_epochs = 80
iterations = 5             # Number of iterations / models
number_of_augmentation = 2 # defines the amount of additional augmentation images of one image
early_stopping = EarlyStopping(monitor='val_loss', patience=5) # Early stopping on val loss - not used

####################
#       Data       #
####################
train_data_dir      = 'data/train/fashion_mnist'
test_data_dir       = 'data/test/fashion_mnist'
classes             = {0: 'T-shirt/top', 
                       1: 'Trouser', 
                       2: 'Pullover', 
                       3: 'Dress', 
                       4: 'Coat',
                       5: 'Sandal', 
                       6: 'Shirt', 
                       7: 'Sneaker', 
                       8: 'Bag', 
                       9: 'Ankle boot'
                      }
num_classes         = len(classes)
classes_fashion     = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                       'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print('Keras version: \t\t%s' % keras.__version__)
print('OpenCV version: \t%s' % cv2.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)

def create_model():
    '''
    Creates a sequential model
    '''
    
    cnn = Sequential()
    
    cnn.add(InputLayer(input_shape=(img_height,img_width,channels)))
    
    # Normalization
    cnn.add(BatchNormalization())
    
    # Conv + Maxpooling
    cnn.add(Convolution2D(64, (4, 4), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(Dropout(0.1))
    
    # Conv + Maxpooling
    cnn.add(Convolution2D(64, (4, 4), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(Dropout(0.3))

    # Converting 3D feature to 1D feature Vektor
    cnn.add(Flatten())

    # Fully Connected Layer
    cnn.add(Dense(256, activation='relu'))

    # Dropout
    cnn.add(Dropout(0.5))
    
    # Fully Connected Layer
    cnn.add(Dense(64, activation='relu'))
    
    # Normalization
    cnn.add(BatchNormalization())

    cnn.add(Dense(num_classes, activation='softmax'))
    cnn.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

    return cnn

create_model().summary()
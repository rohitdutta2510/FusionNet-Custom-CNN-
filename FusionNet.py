import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten, Activation, MaxPooling2D, Add, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt 
import cv2 


def identity_block(X, f, filters):

    # Retrieve Filters
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, s = 1):
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def dense_layer(X, k=32):
    
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    X_shortcut = X
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid',  kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), padding='same',  kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = concatenate([X_shortcut, X], axis=3)
    
    return X


def dense_block(X, dense_num = 6):
    
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    for l in range(dense_num):
        X = dense_layer(X)
    
    return X




kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_1x1 = BatchNormalization(axis = 3)(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = BatchNormalization(axis = 3)(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
    conv_3x3 = BatchNormalization(axis = 3)(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = BatchNormalization(axis = 3)(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
    conv_5x5 = BatchNormalization(axis = 3)(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
    pool_proj = BatchNormalization(axis = 3)(pool_proj)
    pool_proj = Activation('relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
    
    return output


 def transition_layer(X, filters=128):
  
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
    
    return X


def mixed_block(X, filters_residual_block, filters_1x1, filters_3x3_reduce, 
                filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, 
                n, s = 1):
  '''
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  n -- no. of blocks
  '''

  res_block = convolutional_block(X, 
                                  f = 3, 
                                  filters = filters_residual_block)
  for i in range(n):
    res_block = identity_block(X = res_block, 
                               f = 3, 
                               filters = filters_residual_block)

  incep_block = inception_module(X,
                     filters_1x1 = filters_1x1,
                     filters_3x3_reduce = filters_3x3_reduce,
                     filters_3x3 = filters_3x3,
                     filters_5x5_reduce = filters_5x5_reduce,
                     filters_5x5 = filters_5x5,
                     filters_pool_proj = filters_pool_proj)
  
  des_block = dense_block(X,
                          dense_num = n*6)
  

  layer_concat = concatenate([res_block, incep_block, des_block], axis = -1)
  output = transition_layer(layer_concat, 
                            filters=n*64)
  
  return output



def model(input_shape = (224, 224, 3), classes = 2):

    X_input = Input(input_shape)
    
    X = BatchNormalization(axis = 3)(X_input)
    X = Activation('relu')(X)
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    X = mixed_block(X, 
                    filters_residual_block = [64,64,256],
                    filters_1x1=64,
                    filters_3x3_reduce=96,
                    filters_3x3=128,
                    filters_5x5_reduce=16,
                    filters_5x5=32,
                    filters_pool_proj=32,
                    n = 1, 
                    s = 1)
    
    X = mixed_block(X, 
                    filters_residual_block = [128,128,512],
                    filters_1x1=64,
                    filters_3x3_reduce=96,
                    filters_3x3=128,
                    filters_5x5_reduce=16,
                    filters_5x5=32,
                    filters_pool_proj=32,
                    n = 2, 
                    s = 1)
    
    X = mixed_block(X, 
                      filters_residual_block = [256,256,512],
                      filters_1x1=64,
                      filters_3x3_reduce=96,
                      filters_3x3=128,
                      filters_5x5_reduce=16,
                      filters_5x5=32,
                      filters_pool_proj=32, 
                      n = 3, 
                      s = 1)
    
    X = Flatten()(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dense(64, activation='relu')(X)

    X = BatchNormalization(axis=-1)(X)
    output = Dense(classes, activation='softmax')(X)

    model = tf.keras.Model(inputs=X_input, outputs=output)

    return model
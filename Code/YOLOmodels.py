#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:00:44 2018

@author: jmj136
"""
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Conv2D, concatenate, Conv3D
from keras.layers import BatchNormalization, Conv2DTranspose
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import ELU
import keras.backend as K

def BuildInceptionModel(input_shape):
    # get inception
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=input_shape)
    
    # add extra layers to be fine-tuned for our network
    x = base_model.get_layer('mixed9').output
    for f in [64,128]:
        x = Conv2D(f,(3,3),padding='valid')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
    final_layer = Conv2D(7,(1,1),padding='valid')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=final_layer)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def YOLOloss(y_pred,y_true):
    grid_shape = [9,9]
#    batch_size = K.int_shape(y_true)[0]
    cell_x = K.tf.to_float(K.reshape(K.tile(K.tf.range(grid_shape[0]), [grid_shape[1]]), (1, grid_shape[0], grid_shape[1], 1, 1)))
    cell_y = K.transpose(cell_x, (0,2,1,3,4))
    cell_grid = K.concatenate([cell_x,cell_y], -1)
    
    
    return loss
    

if __name__ == '__main__':
    testmodel = BuildInceptionModel((490,490,3))
    testmodel.summary()

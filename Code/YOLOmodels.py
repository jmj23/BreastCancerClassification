#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:00:44 2018

@author: jmj136
"""
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.layers import Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import ELU
import keras.backend as K

def BuildInceptionModel(input_shape):
    # get inception V3 network and weights
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(*input_shape[:2],3))
    
    # add extra layers to be fine-tuned for our network
    x = base_model.get_layer('mixed7').output
    for f in [64,128]:
        x = Conv2D(f,(3,3),padding='valid')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
    # final output layer
    final_layer = Conv2D(7,(1,1),padding='valid')(x)
    # make temporary model
    tempmodel = Model(inputs=base_model.input,outputs=final_layer)
    # remove input layer and first convolutional layer
    tempmodel.layers.pop(0)
    tempmodel.layers.pop(0)
    # make convolution layer    
    inp = Input(shape=input_shape)
    x = Conv2D(3,(3,3),padding='same',)(inp)
    # call model on these new layers
    new_output = tempmodel(x)
    # this is the model we will train
    model = Model(inputs=inp, outputs=new_output)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def YOLOloss(y_pred,y_true):
    # coordinate losses
    pred_xy = K.sigmoid(y_pred[...,:2])
    true_xy = K.sigmoid(y_true[...,:2])
    pred_wh = K.exp(y_pred[...,2:4])
    true_wh = K.exp(y_true[...,2:4])
    coord_loss = K.mean(K.square(pred_xy-true_xy)+K.square(pred_wh-true_wh))
    # object loss
    pred_obj = K.sigmoid(y_pred[...,4])
    true_obj = K.sigmoid(y_true[...,4])
    obj_loss = K.sum(-(1-true_obj)*K.log(1-pred_obj)-true_obj*K.log(pred_obj))
    # class loss
    pred_class = y_pred[...,5:7]
    true_class = y_true[...,5:7]
    class_loss = K.categorical_crossentropy(true_class,pred_class,from_logits=True)
    
    # weighted losses
    loss = coord_loss + 5*obj_loss + class_loss
    
    return loss
    

if __name__ == '__main__':
    testmodel = BuildInceptionModel((384,384,5))
    testmodel.summary()

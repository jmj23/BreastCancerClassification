#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:00:44 2018

@author: jmj136
"""
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization, Conv2DTranspose
from keras.layers.advanced_activations import ELU
import keras.backend as K

#%%
def BuildInceptionModel(input_shape):
    # get inception V3 network and weights
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(*input_shape[:2],3))
    
    # add extra layers to be fine-tuned for our network
    x = base_model.get_layer('mixed7').output
    for f in [64,64,64,64]:
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

#%%
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

#%%
def CoordsToTarget(coords,dim,grid_size):
    target = np.zeros((*grid_size,7))
    for r in [0,5,10,15]:
        cdat = coords[r:r+5]
        if np.any(cdat):
            # coordinates are in format:
            # [x1,y1,x2,y2,b/m]
            # calculate center coordinates
            bx = (cdat[2]+cdat[0])/2
            by = (cdat[3]+cdat[1])/2
            # calculate width and height
            bw = cdat[2]-cdat[0]
            bh = cdat[3]-cdat[1]
            # calculate grid square width and height
            gw = dim[0]/grid_size[0]
            gh = dim[1]/grid_size[1]
            # find grid square that contains center
            xind = np.floor(bx/gw).astype(np.int)
            yind = np.floor(by/gh).astype(np.int)
            # find offset from grid square
            cx = gw*xind
            cy = gh*yind
            # calculate relative offset within square
            ox = (bx-cx)/gw
            oy = (by-cy)/gh
            # calculate center logits
            tx = np.log(ox/(1-ox))
            ty = np.log(oy/(1-oy))
            # calculate height/width logits
            tw = np.log(bw/gw)
            th = np.log(bh/gh)
            # class assignment
            pb = np.int(cdat[4]==1)
            pm = np.int(cdat[4]==2)
            # combine into vector
            vec = np.array([tx,ty,tw,th,1,pb,pm])
            # assign to target grid
            target[xind,yind,:] = vec
    return target

#%%
def CollectYOLO(targ,gw,gh,xi,yi):
    # get center x and y
    bx = gw / (1 + np.exp(-targ[0])) + xi*gw
    by = gh / (1 + np.exp(-targ[1])) + yi*gh
    # get width and height
    bw = gw*np.exp(targ[2])
    bh = gh*np.exp(targ[3])
    # convert center to corner x,y
    bx -= bw/2
    by -= bh/2
    # get class
    malig = np.float(targ[6]>targ[5])
    return bx,by,bw,bh,malig

def TargetToCoords(target,im_dim,conf=.5):
    # look for objects
    (xinds,yinds) = np.where(target[...,4]>conf)
    # get grid parameters
    nx = target.shape[0]
    ny = target.shape[1]
    gw = im_dim[0]/nx
    gh = im_dim[1]/ny
    # loop over all objects detected
    roi_dat = [CollectYOLO(target[xi,yi,:],gw,gh,xi,yi) for xi,yi in zip(xinds,yinds)]
    return roi_dat
    
    

#%%
import matplotlib.patches as patches
class YOLOcallback(keras.callbacks.Callback):
    
    def __init__(self, image, model):
        self.image = image
        self.im = image[...,1]
        self.model = model
        
    def ConvertYOLOtoCoords(self,target,imdim,conf):
        # look for objects
        (xinds,yinds) = np.where(target[...,4]>conf)
        # get grid parameters
        nx = target.shape[0]
        ny = target.shape[1]
        gw = imdim[0]/nx
        gh = imdim[1]/ny
        # loop over all objects detected
        roi_dat = [CollectYOLO(target[xi,yi,:],gw,gh,xi,yi) for xi,yi in zip(xinds,yinds)]
        return roi_dat
    
    def UpdatePatches(self,remove=True):
        # get prediction on image
        pred = self.model.predict(self.image[np.newaxis,...],batch_size=1)[0]
        # get coordinates from YOLO target
        roi_xywh = self.ConvertYOLOtoCoords(pred,self.image.shape,.5)
        
        # remove current patches
        if remove:
            for rect in self.rects:
                rect.remove()
        # create rectangle patches
        self.rects = [patches.Rectangle((c[0],c[1]),c[2],c[3],
                                   linewidth=1,
                                   edgecolor='r' if c[4]==1 else 'g',
                                   facecolor='none') for c in roi_xywh]
        # Add the patches to the Axes
        for rect in self.rects:
            self.ax.add_patch(rect)
        plt.pause(.01)
        plt.draw()
        
    def on_train_begin(self, logs={}):
        fig, ax = plt.subplots()
        self.ax = ax
        self.ax.imshow(self.im,cmap='gray',
                  vmin=np.min(self.im),
                  vmax=np.max(self.im))
        self.ax.set_axis_off()
        # Create patches
        self.UpdatePatches(False)
    
    def on_epoch_end(self, epoch, logs={}):
        self.UpdatePatches(True)
        return
    
    def on_batch_end(self, batch, logs={}):
        self.UpdatePatches(True)
        return


if __name__ == '__main__':
    testmodel = BuildInceptionModel((384,384,5))
    testmodel.summary()

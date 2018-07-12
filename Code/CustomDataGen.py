#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:51:29 2018

"""

import numpy as np
import keras
import os 

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(256,256), 
                 n_channels=1, shuffle=True,grid_size=(18,18)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.grid_size = grid_size
        self.data_path = os.path.expanduser(
                            os.path.join('~','r-fcb-isilon',
                                     'groups','StrigelGroup',
                                     'BreastCancerClassification',
                                     'ProcessedData',''))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.grid_size, 7))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load image slice
            im = np.load(self.data_path + ID + '.npy')
            # load coordinates
            coords = np.loadtxt(self.data_path + ID + '.txt')
            # convert coordinates to YOLO
            # preallocate
            target = np.zeros((*self.grid_size,7))
            # iterate over 4 possible ROIs
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
                    gw = self.dim[0]/self.grid_size[0]
                    gh = self.dim[1]/self.grid_size[1]
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
                    vec = [tx,ty,tw,th,1,pb,pm]
                    # assign to target grid
                    target[xind,yind,:] = vec
            
            # Store image in batch
            X[i,] = im

            # Store target in batch
            y[i,] = target

        return X, y
    
    def __get_input(self,filepath):
        return np.load(filepath)
    
    def __get_target(self,filepath):
        return np.loadtxt(filepath)
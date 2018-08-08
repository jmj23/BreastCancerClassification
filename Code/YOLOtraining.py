#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:07:12 2018

@author: jmj136
"""

import sys
import os
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import time
from glob import glob
from VisTools import YOLOviewer
from CustomDataGen import DataGenerator
from YOLOtools import CoordsToTarget, BuildInceptionModel
from YOLOtools import YOLOloss, YOLOcallback

# Use first available GPU
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

np.random.seed(seed=1)
    
#%% Some Setup

# Model Save Path/name
timestr = time.strftime("%Y%m%d%H%M")
model_filepath = os.path.join('Models','BreastLesionDetectionModel_{}.h5'.format(timestr))

# Data path
data_dir = '/data/jmj136/LesionDetection/ProcessedData/'

im_dim = (384,384)
grid_dim = (18,18)
batch_size = 16
num_epochs = 30
workers=8

#%% Data setup
if not 'train_IDs' in locals():
    print('Setting up data...')
    # Get list of data
    all_files = glob(os.path.join(data_dir,'*.npy'))
    IDs = [os.path.split(os.path.splitext(f)[0])[1] for f in all_files]
    # find which are malignant
    mal_inds = ['M' in i for i in IDs]
    # find which are benign
    ben_inds = ['B' in i for i in IDs]
    
    # Use only slices that contain a lesion
    from itertools import compress
    IDs = list(compress(IDs,[a or b for a, b in zip(mal_inds, ben_inds)]))
    print('Found {} slices'.format(len(IDs)))
    
    # Get list of subjects
    subjs = [i[:3] for i in IDs]
    unq_subjs = np.unique(subjs)
    # Random choice for validation subjects
    val_frac = .2
    num_subjs = len(unq_subjs)
    val_subjs = np.random.choice(unq_subjs,np.int(val_frac*num_subjs),replace=False)
    val_inds = [s in val_subjs for s in subjs]
    train_inds = [s not in val_subjs for s in subjs]
    
    # assign training and validation classes
    val_IDs = list(compress(IDs,val_inds))
    train_IDs = list(compress(IDs,train_inds))
    print('Using {} slices for training'.format(len(train_IDs)))
    print('Using {} slices for validation'.format(len(val_IDs)))

# load in validation data
if not 'y_val' in locals():
    print('Loading validation data')
    x_val = np.zeros((len(val_IDs),*im_dim,5))
    for i,ID in enumerate(val_IDs):
        im = np.load(data_dir + ID + '.npy')
        x_val[i] = im
    y_val = np.zeros((len(val_IDs),*grid_dim,7))
    for i,ID in enumerate(val_IDs):
        # load coordinates
        coords = np.loadtxt(data_dir + ID + '.txt')
        # convert coordinates to YOLO
        target = CoordsToTarget(coords,im_dim,grid_dim)
        # store
        y_val[i] = target

#%% Setup model and training
print('Setting up model...')

# Generators
train_gen = DataGenerator(train_IDs, batch_size=batch_size, dim=im_dim,
                          n_channels=5, shuffle=True,grid_size=grid_dim)
# number of steps
steps = np.int(len(train_IDs)/batch_size)

# Build model
DetectModel = BuildInceptionModel((*im_dim,5))
# Compile with custom loss
opt = keras.optimizers.Adam()
loss = YOLOloss
DetectModel.compile(opt,loss)

# Callbacks
cp = ModelCheckpoint(model_filepath,monitor='val_loss',verbose=0,
                     save_best_only=True,save_weights_only=True)
CBs = [cp,YOLOcallback(x_val[15],DetectModel)]
#%% Train model on dataset
print('Starting training...')
DetectModel.fit_generator(generator=train_gen,
                    validation_data=(x_val,y_val),
                    steps_per_epoch=steps,
                    epochs=num_epochs,
                    use_multiprocessing=True,
                    workers=workers,
                    callbacks=CBs)

#%% Test some results
# load best weights
DetectModel.load_weights(model_filepath)
val_predict = DetectModel.predict(x_val,batch_size=32)
disp_ind = 50
YOLOviewer(x_val[disp_ind,...,2],val_predict[disp_ind,...],conf=.5)
YOLOviewer(x_val[disp_ind,...,2],y_val[disp_ind,...],conf=.5)

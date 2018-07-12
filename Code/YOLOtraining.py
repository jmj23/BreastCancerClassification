#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:07:12 2018

@author: jmj136
"""

import sys
import os
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from matplotlib import pyplot as plt
import numpy as np
import keras
import time
from glob import glob
import YOLOmodels
from VisTools import YOLOviewer
from CustomDataGen import DataGenerator

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
data_dir = os.path.expanduser(os.path.join('~','r-fcb-isilon','groups',
                                           'StrigelGroup',
                                           'BreastCancerClassification',
                                           'ProcessedData',''))
#%% Data setup
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

#%% Setup model and training
print('Setting up model...')
# Generators
train_gen = DataGenerator(train_IDs, batch_size=32, dim=(256,256),
                          n_channels=1, shuffle=True,grid_size=(18,18))
val_gen = DataGenerator(val_IDs, batch_size=32, dim=(256,256),
                        n_channels=1, shuffle=True,grid_size=(18,18))

# Build model
DetectModel = YOLOmodels.BuildInceptionModel((384,384,5))
# Compile with custom loss
opt = keras.optimizers.Adam()
loss = YOLOmodels.YOLOloss
DetectModel.compile(opt,loss)

# Callbacks

#%% Train model on dataset
print('Starting training...')
DetectModel.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    steps_per_epoch=50,
                    validation_steps=13,
                    epochs=2,
                    use_multiprocessing=True,
                    workers=6)
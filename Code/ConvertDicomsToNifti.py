#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:25:23 2018

@author: jmj136
"""
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
import numpy as np
import pydicom as dcm
import nibabel as nib
import os
import glob
from natsort import natsorted
from operator import itemgetter
data.sort(key=itemgetter(1))


# output directory
output_dir = os.path.join('/','home','jmj136','r-fcb-isilon','groups','StrigelGroup',
                         'BreastCancerClassification','NIFTIs')
# RMRLHBLC001\MR\20120712\MRI BREAST BILATERAL W AND OR W O CONTRAST
# Dataset directory
base_path = os.path.join('/','home','jmj136','r-fcb-isilon','groups','StrigelGroup',
                         'BreastCancerClassification','Datasets')
# list of subject directories, sorted by number
subj_dirs = natsorted(glob.glob(os.path.join(base_path, "*", "")))
# sort naturally

# append '/MR' directory to each
mr_dirs = [os.path.join(dir,'MR') for dir in subj_dirs]

# current directory
cur_dir = mr_dirs[0]
# find date dir
cur_dir = glob.glob(os.path.join(cur_dir, "*", ""))[0]
# find main dicom dir
cur_dir = glob.glob(os.path.join(cur_dir, "*", ""))[0]
# find various dicom dirs
dcm_dirs = natsorted(glob.glob(os.path.join(cur_dir, "*", "")))
# find T2fat dicom dir
T2dir = [string for string in dcm_dirs if 'T2fat' in string][-1]
# find pre contrast dir
pre_dir = [string for string in dcm_dirs if 'Pre Ax3d' in string][-1]
# find post-contrast dir
post_dir = [string for string in dcm_dirs if 'Dur Ax3d' in string][-1]

# Load in dicoms
def LoadDicomDir(directory,post_con=False):
    # get file list
    FNs = sorted(glob.glob(os.path.join(directory,'*.dcm')))
    # load data
    dicms = [dcm.read_file(fn) for fn in FNs]
    # prepare sorting list of tuples
    FNinds = list(range(len(FNs)))
    locs = [(float(dicm.SliceLocation)) for dicm in dicms]
    times = [(int(dicm.AcquisitionTime)) for dicm in dicms]
    loctimes = list(zip(locs,times,FNinds))
    # sort by time then slice
    loctimes.sort(key=itemgetter(0))
    loctimes.sort(key=itemgetter(1))
    # extract sorting indices
    sort_inds = [ele[2] for ele in loctimes]
    
    # reference image
    RefDs = dicms[0]
    # Load dimensions based on the number of rows, columns, and slices
    volsize = (int(RefDs.Rows), int(RefDs.Columns), len(FNs))
    # The array is sized based on volsize
    ims = np.zeros(volsize, dtype=RefDs.pixel_array.dtype)
    
    # loop through all the DICOM files
    for ind in range(len(sort_inds)):
        arg = sort_inds[ind]
        # store the raw image data
        ims[:, :, ind] = np.abs(dicms[arg].pixel_array)
    ims = np.rollaxis(ims,2,0)
    if post_con:
        phase1,phase2,phase3,phase4 = np.split(ims,4)
        return_ims = np.stack((phase1,phase2,phase3,phase4),axis=-1)
    else:
        return_ims = ims[...,np.newaxis]
    return return_ims

T2_ims = LoadDicomDir(T2dir)
pre_ims = LoadDicomDir(pre_dir)
post_ims = LoadDicomDir(post_dir,post_con=True)

from VisTools import multi_slice_viewer0, slice_viewer4D
multi_slice_viewer0(T2_ims[...,0]/1000)
multi_slice_viewer0(pre_ims[...,0]/1000)
slice_viewer4D(post_ims/1000)
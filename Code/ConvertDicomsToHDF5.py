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
import h5py
import os
import glob
from natsort import natsorted
from operator import itemgetter

# output directory
output_path = os.path.join('/','home','jmj136','r-fcb-isilon','groups','StrigelGroup',
                         'BreastCancerClassification','RawHDF5','subj_{}.hdf5')
# Dataset directory
base_path = os.path.join('/','home','jmj136','r-fcb-isilon','groups','StrigelGroup',
                         'BreastCancerClassification','Datasets')

# Function for finding the specific DICOM directories for each patient
def FindDirectories(mr_dir):
    # find date dir
    cur_dir = glob.glob(os.path.join(mr_dir, "*", ""))[0]
    # find main dicom dir
    cur_dir = glob.glob(os.path.join(cur_dir, "*", ""))[0]
    # find various dicom dirs
    dcm_dirs = natsorted(glob.glob(os.path.join(cur_dir, "*", "")))
    try:
        # find T2fat dicom dir
        T2dir = [string for string in dcm_dirs if 'T2fat' in string][-1]
        # find pre contrast dir
        pre_dir = [string for string in dcm_dirs if 'Pre Ax3d' in string and not 'RECON' in string][-1]
        # find post-contrast dir
        post_dir = [string for string in dcm_dirs if 'Dur Ax3d' in string and not 'RECON' in string][-1]
    except IndexError as e:
        T2dir = False
        pre_dir = False
        post_dir = False
    return T2dir,pre_dir,post_dir

# function for creating transformation matrix
def CreateTmatrix(info):
    x_basis = info.ImageOrientationPatient[:3]
    x_basis = np.array([int(s) for s in x_basis])
    y_basis = info.ImageOrientationPatient[3:6]
    y_basis = np.array([int(s) for s in y_basis])
    z_basis = np.cross(x_basis,y_basis)
    delta_i = float(info.PixelSpacing[0])
    delta_j = float(info.PixelSpacing[1])
    delta_k = float(info.SpacingBetweenSlices)
    origin = info.ImagePositionPatient
    origin = np.array([float(s) for s in origin])
    tmat = np.c_[x_basis*delta_i,y_basis*delta_j,z_basis*delta_k,origin]
    tmat = np.vstack((tmat,np.array([0,0,0,1])))
    return tmat

# Function for loading in dicoms from a given directory
def LoadDicomDir(directory):
    # get file list
    FNs = sorted(glob.glob(os.path.join(directory,'*.dcm')))
    # load data
    dicms = [dcm.read_file(fn) for fn in FNs]
    # prepare sorting list of tuples
    FNinds = list(range(len(FNs)))
    locs = [(float(dicm.SliceLocation)) for dicm in dicms]
    times = [(int(dicm.AcquisitionTime)) for dicm in dicms]
    # determine if multiple phases in this directory
    post_con = np.unique(times).size>1 or len(times)>200
    # zip up slice locations, acquisition times, and file indicies
    loctimes = list(zip(locs,times,FNinds))
    # sort by time then slice
    loctimes.sort(key=itemgetter(0))
    loctimes.sort(key=itemgetter(1))
    # extract sorting indices
    sort_inds = [ele[2] for ele in loctimes]
    
    # reference image for sizing
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
        return_ims = np.stack((phase1,phase2,phase3,phase4),axis=-1).astype(np.float)
    else:
        return_ims = ims[...,np.newaxis].astype(np.float)
        
        
    # find slice extrema
    zmin = np.min(np.array(locs))
    zmax = np.max(np.array(locs))
    zrange = np.array([zmin,zmax])
    
    # create transformation matrix
    tMat = CreateTmatrix(RefDs)
    
    return return_ims,zrange,tMat

# get list of subject directories, sorted by number
subj_dirs = natsorted(glob.glob(os.path.join(base_path, "*", "")))
subj_nums = [string[-4:-1] for string in subj_dirs]
# append '/MR' directory to each
mr_dirs = [os.path.join(dir,'MR') for dir in subj_dirs]

# loop over all subjects
sag_subjs = []
subj_range = range(0,len(mr_dirs)) 
subj_range = [371]
for subj in subj_range:
    # current directory
    cur_dir = mr_dirs[subj]
    cur_subj = subj_nums[subj]
    _,pre_dir,post_dir = FindDirectories(cur_dir)
    if not pre_dir:
        print('Skipping subject',cur_subj)
        sag_subjs.append(subj)
    else:
        print('Processing subject',cur_subj,'...')
        #T2_ims = LoadDicomDir(T2dir)
        nonfat_ims,nonfat_zrange,nonfat_tmat = LoadDicomDir(pre_dir)
        dyn_ims, dyn_zrange,dyn_tmat = LoadDicomDir(post_dir)

        
        # register T2 to others
        #T2_img = ants.from_numpy(T2_ims[...,0].astype(np.float))
        #T2_img_resamp = T2_img.resample_image((124,512,512),use_voxels=True,interp_type=0)
        #pre_img = ants.from_numpy(pre_ims[...,0].astype(np.float))
        #T2_tx = ants.registration(fixed=pre_img,moving=T2_img,type_of_transform='Similarity',aff_metric='mattes')
        #reg_T2 = T2_tx['warpedmovout']
        #T2_ims_reg = reg_T2.numpy()[...,np.newaxis]
        #import ants
        
        # export to .hdf5 file
        savepath = output_path.format(cur_subj)
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset("nonfat_images",  data=nonfat_ims, dtype='f')
            hf.create_dataset("nonfat_zrange", data=nonfat_zrange, dtype='f')
            hf.create_dataset("nonfat_tmat", data=nonfat_tmat, dtype='f')
            hf.create_dataset("dyn_images", data=dyn_ims,dtype='f')
            hf.create_dataset("dyn_zrange", data=dyn_zrange, dtype='f')
            hf.create_dataset("dyn_tmat", data=dyn_tmat, dtype='f')

print('Done')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:11:26 2018

@author: jmj136
"""

import pydicom as dcm
import glob
import h5py
import numpy as np
import os
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from VisTools import MultiROIviewer
from natsort import natsorted
import cv2

im_reshape = (384,384)

isilon_dir = os.path.expanduser('~/r-fcb-isilon')

ROIdir = os.path.join(isilon_dir,'groups/StrigelGroup/BreastCancerClassification/ROIs/*RTSS')
gen_ROI_dir = os.path.join(isilon_dir,'groups/StrigelGroup/BreastCancerClassification/ROIs/RMRLHBLC{:03d}*')
image_path = os.path.join(isilon_dir,'groups','StrigelGroup','BreastCancerClassification',
                          'RawHDF5','subj_{}.hdf5')


# extract coordinates from a contour
def ExtractCoords(contour):
    contour_data = np.array([float(dat) for dat in contour.ContourData])
    contour_array = np.reshape(contour_data,(-1,3))
    z = contour_array[0,2]
    xmin = np.min(contour_array[:,0])
    xmax = np.max(contour_array[:,0])
    ymin = np.min(contour_array[:,1])
    ymax = np.max(contour_array[:,1])
    return [z,xmin,xmax,ymin,ymax]

def LoadImageData(subj):
    cur_path = image_path.format(subj)
    with h5py.File(cur_path, 'r') as hf:
        nonfat_ims = np.array(hf.get("nonfat_images"))
        nonfat_zrange = np.array(hf.get("nonfat_zrange"))
        nonfat_tmat = np.array(hf.get("nonfat_tmat"))
        dyn_ims = np.array(hf.get("dyn_images"))
        dyn_zrange = np.array(hf.get("dyn_zrange"))
        dyn_tmat = np.array(hf.get("dyn_tmat"))
    nonfat = [nonfat_ims,nonfat_zrange,nonfat_tmat]
    dyn = [dyn_ims,dyn_zrange,dyn_tmat]
    return nonfat,dyn

def ConvertCoords(tmat,coord):
    xyz1 = np.array([coord[1],coord[3],coord[0],1])
    xyz2 = np.array([coord[2],coord[4],coord[0],1])
    coord1 = np.linalg.solve(tmat,xyz1)[:-1]
    coord2 = np.linalg.solve(tmat,xyz2)[:-1]
    return coord1,coord2

# all roi files
roifiles = natsorted(glob.glob(ROIdir))
# get list of subjects
subjs = np.unique([int(fp[86:89]) for fp in roifiles])

# ~loop over subjects~
# current subject
cur_subj = subjs[-6]
print('Processing subject',cur_subj)
# get all file paths
cur_roi_files = glob.glob(gen_ROI_dir.format(cur_subj))
# get subject number
subj_num = os.path.split(cur_roi_files[0])[1][8:11]

# load in images and transform data
print('Loading images...')
nonfat,dyn = LoadImageData(subj_num)
if not np.array_equal(dyn[0].shape[:3],nonfat[0].shape[:3]):
    raise ValueError('Images are not equal size')
    
# rescale images
print('Resizing images...')
nonfat_ims = np.zeros((nonfat[0].shape[0],im_reshape[0],im_reshape[1],nonfat[0].shape[-1]))
dyn_ims = np.zeros((dyn[0].shape[0],im_reshape[0],im_reshape[1],dyn[0].shape[-1]))
for ii in range(nonfat_ims.shape[0]):
    nonfat_ims[ii,...,0] = cv2.resize(nonfat[0][ii,:,:,0],im_reshape)
    for cc in range(dyn_ims.shape[-1]):
        dyn_ims[ii,...,cc] = cv2.resize(dyn[0][ii,:,:,cc],im_reshape)
    
# normalize and combine images
print('Normalizing images...')
for im in nonfat_ims:
    im /= np.max(im)
for im in dyn_ims:
    im /= np.max(im)
comb_ims = np.concatenate((nonfat_ims,dyn_ims),axis=-1)

# preallocate coordinate array for this subject
# coordinates are stored in form [slice,coords]
# where coords is a vector length 20 that contains
# sets of [x1,y1,x2,y2,bm] and zeros, depending on the number
# of ROIs on that are contained on that slice
coord_array = np.zeros((comb_ims.shape[0],20))
print('Processing ROIs')
# ~loop over ROIs~
for roi_num in range(len(cur_roi_files)):
    # current ROI
    cur_fp = cur_roi_files[roi_num]
    
    # get ROI data
    roi_data = dcm.read_file(cur_fp)
    contour_seq = roi_data.ROIContourSequence[0].ContourSequence
    if len(roi_data.ROIContourSequence)>1:
        raise ValueError('More than 1 contour sequence')
    all_coords = np.array([ExtractCoords(cont) for cont in contour_seq])
    all_coords = all_coords[all_coords[:,0].argsort()]
    
    # benign/malignant
    is_malignant = roi_data.StructureSetLabel=='m'
    
    # convert coordinates
    tmat = nonfat[2]
    img_coords = [ConvertCoords(tmat,c) for c in all_coords]
    zrange = np.array([np.min(all_coords[:,0]),np.max(all_coords[:,0])])
    img_zrange = np.ceil((zrange-nonfat[1][0])/(nonfat[1][1]-nonfat[1][0])*nonfat[0].shape[0]).astype(np.int)
    img_zs = np.arange(img_zrange[0],img_zrange[1],dtype=np.int)
    # coordinates in the form of [x1,y1,x2,y2]
    coords = np.r_[img_coords[0][1][:2],img_coords[0][0][:2]]
    
    # transform coordinates to reshaped images
    mult = nonfat[0].shape[1]/im_reshape[0]
    coords /= mult
    # add m/b to end: 1 is benign, 2 is malignant
    coords = np.r_[coords,np.float(is_malignant)+1]
    # add to coordinate array
    for ss in range(img_zs.shape[0]):
        coord_array[img_zs[ss],(5*roi_num):(5*roi_num+5)] = coords
    
MultiROIviewer(comb_ims[...,3],coord_array)

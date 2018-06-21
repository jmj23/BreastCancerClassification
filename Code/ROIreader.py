#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:11:26 2018

@author: jmj136
"""

import pydicom as dcm
import glob
import numpy as np
import os
isilon_dir = os.path.expanduser('~/r-fcb-isilon')

ROIdir = os.path.join(isilon_dir,'groups/StrigelGroup/BreastCancerClassification/ROIs/*RTSS')


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



roifiles = glob.glob(ROIdir)

cur_fp = roifiles[0]
# get file name
_,cur_fn = os.path.split(cur_fp)
# get subject number
subj_num = cur_fn[8:11]
# get ROI data
roi_data = dcm.read_file(cur_fp)
contour_seq = roi_data.ROIContourSequence[0].ContourSequence
coord_array = np.array([ExtractCoords(cont) for cont in contour_seq])
coord_array = coord_array[coord_array[:,0].argsort()]


ROIlist = []
for file in roifiles:
    roi = dcm.read_file(file)
    ROIlist.append(roi)

# Want to convert each of these ROIs into lists of 4x3 array 
# of x,y,slice coordinates in pixel space, i.e.
# [x1, y1, slice    upper-left corner
# x2, y2, slice     upper-right corner
# x3, y3, slice     lower-right corner
# x4, y4, slice]    lower-left corner
# for each ROI on each slice, i.e.
# [[slice_1_array_1,slice_1_array_2],   2 ROIs on this slice
#  [slice_2_array_1],                   1 ROI on this slice
#  [slice_3_array_1,slice_3_array_2]]   2 ROIs on this slice
# and save them with patient identifier too
# to be able to match them with images
# Finally, a tag for B/M
    


def ConvertROI(roi):
    fields = roi.desired_fields
    coords = np.array([fields[0],fields[1]]) # create array described above
    patient_num = roi.patient_num           # some patient identifier for matching to images
    malig = roi.is_malignant                # false if benign, true if malignant
    return coords,patient_num,malig



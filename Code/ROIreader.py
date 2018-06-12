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

ROIdir = '~/r-fcb-isilon/groups/StrigelGroup/BreastCancerClassification/ROIs/{}'

globfiles = glob.glob('../ROIs/*RTSS')
FNs = [os.path.split(file)[1] for file in globfiles]
FPs = [ROIdir.format(fn) for fn in FNs]


ROIlist = []
for file in FPs:
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
    
# To be run on each ROI on each slice
def ConvertROI(roi):
    fields = roi.desired_fields
    coords = np.array([fields[0],fields[1]]) # create array described above
    patient_num = roi.patient_num           # some patient identifier for matching to images
    malig = roi.is_malignant                # false if benign, true if malignant
    return coords,patient_num,malig



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:59:09 2018

@author: jmj136
"""
import numpy as np

# coords come in form:
# [x1, y1, slice    upper-left corner
# x2, y2, slice     upper-right corner
# x3, y3, slice     lower-right corner
# x4, y4, slice]    lower-left corner
# need them to be in form
# tx, ty, tw, th
def CoordToGrid(coords,grid_dims):
    # calculate center
    center = np.mean(coords,axis=0)
    # convert to grid coordinates
    [bx,by] = center[0:2]/grid_dims
    grid_inds = np.floor([bx,by]).astype(np.int)
    # calculate width and height
    iw = coords[1,0]-coords[0,0]
    ih = coords[2,1]-coords[0,1]
    # convert to grid coordinates
    bw = iw/grid_dims[0]
    bh = ih/grid_dims[1]
    return grid_inds,[bx,by,bw,bh]

# image and grid parameters
im_dims = np.array([256,256])
grid_size = np.array([9,9])
grid_dims = im_dims/grid_size

# test coordinates
test_coords = np.array([[15,15,0],
                   [30,15,0],
                   [30,30,0],
                   [15,30,0]])

# pre-allocate array for current slice
cur_array = np.zeros((1,im_dims[0],im_dims[1],4))
# loop over all ROIs for current slice
# get grid coordinates
inds,grid_coords = CoordToGrid(test_coords,grid_dims)
# update solution array
cur_array[0,inds[0],inds[1],:] = grid_coords
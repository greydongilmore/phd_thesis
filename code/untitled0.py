#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:56:33 2022

@author: greydon
"""

fixed_orig =  slicer.util.arrayFromMarkupsControlPoints(getNode('frameSystemFiducials'))
moving_orig = slicer.util.arrayFromMarkupsControlPoints(getNode('sub-P150_desc-fiducials_fids'))


fixed_orig = fixed_orig.T
moving_orig = moving_orig.T

Ncoords, Npoints = fixed_orig.shape
Ncoords_Y, Npoints_Y = moving_orig.shape

Xbar = np.mean(fixed_orig,1)
Ybar = np.mean(moving_orig,1)

Xtilde = fixed_orig-np.tile(Xbar,(1,Npoints)).reshape(fixed_orig.shape)
Ytilde = moving_orig-np.tile(Ybar,(1,Npoints_Y)).reshape(moving_orig.shape)
H = moving_orig @ np.transpose(moving_orig)

U, S, V= np.linalg.svd(H)

VU = np.matmul(V.transpose(), U)
detVU = np.linalg.det(VU)
diag = np.eye(3, 3)
diag[2][2] = np.linalg.det(VU)
X = np.matmul(V.transpose(), np.matmul(diag, U.transpose()))


R = V*np.diag(np.c_[1, 1, np.linalg.det(V@U)])*U.T
t = Ybar - R@Xbar
FREvect = R@fixed_orig + np.tile(t,(1,Npoints)).reshape(3,801) - moving_orig
FRE = np.sqrt(np.mean(np.sum(FREvect**2,0)))

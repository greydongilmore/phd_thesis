#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 08:11:06 2022

@author: greydon
"""

import numpy as np
import math

def AOPA_Major(X, Y, tol):
	"""
	Computes the Procrustean fiducial registration between X and Y with 
	anisotropic Scaling:
		
		Y = R * A * X + t
		
	where X is a mxn matrix, m is typically 3
		  Y is a mxn matrix, same dimension as X
		  
		  R is a mxm rotation matrix,
		  A is a mxm diagonal scaling matrix, and
		  t is a mx1 translation vector
		  
	based on the Majorization Principle

	Elvis C.S. Chen
	chene AT robarts DOT ca
	"""
	[m,n] = X.shape
	II = np.identity(n) - np.ones((n,n))/n
	mX = np.nan_to_num(np.matmul(X,II)/np.linalg.norm(np.matmul(X,II), ord=2, axis=1, keepdims=True))
	mY = np.matmul(Y,II)
	
	# estimate the initial rotation
	B = np.matmul(mY, mX.transpose())
	u, s, vh = np.linalg.svd(B)
	
	# check for flip
	D = np.identity(m)
	D[m-1,m-1] = np.linalg.det(np.matmul(u,vh))
	R = np.matmul(u, np.matmul(D,vh))
	
	# loop
	err = np.Infinity
	E_old = 1000000 * np.ones((m,n))
	while err > tol: 
		u, s, vh = np.linalg.svd( np.matmul(B, np.diag(np.diag(np.matmul(R.transpose(),B)))) )
		D[m-1,m-1] = np.linalg.det(np.matmul(u,vh))
		R = np.matmul(u, np.matmul(D,vh))
		E = np.matmul(R,mX) - mY
		err = np.linalg.norm(E-E_old)
		E_old = E
	# after rotation is computed, compute the scale
	B = np.matmul(Y, np.matmul(II, X.transpose()))
	A = np.diag( np.divide( np.diag( np.matmul(B.transpose(), R)), np.diag( np.matmul(X, np.matmul(II, X.transpose()))) ) )
	if (math.isnan(A[2,2])):
		# special case for ultrasound calibration, where z=0
		A[2,2] = .5 * (A[0,0] + A[1,1]) # artificially assign a number to the scale in z-axis
	# calculate translation
	t = np.reshape(np.mean( Y - np.matmul(R, np.matmul(A,X)), 1), [m,1])
	return[R,t,A]

def p2l(X, Y, D, tol):
	"""
	Computes the Procrustean point-line registration between X and Y+nD with 
	anisotropic Scaling,
		
	where X is a mxn matrix, m is typically 3
		  Y is a mxn matrix denoting line origin, same dimension as X
		  D is a mxn normalized matrix denoting line direction
		  
		  R is a mxm rotation matrix,
		  A is a mxm diagonal scaling matrix, and
		  t is a mx1 translation vector
		  Q is a mxn fiducial on line that is closest to X after registration
		  fre is the fiducial localization error
		  
	based on the Majorization Principle
	"""
	[m,n] = X.shape
	err = np.Infinity
	E_old = 1000000 * np.ones((m,n))
	e = np.ones((1,n))
	# intialization
	Q = Y
	# normalize the line orientation just in case
	Dir = D/np.linalg.norm(D, ord=2,axis=0,keepdims=True)
	while err > tol:
		[R, t, A] = AOPA_Major(X, Q, tol)
		E  = Q-np.matmul(R,np.matmul(A,X))-np.matmul(t,e)
		# project point to line
		Q = Y+Dir*np.tile(np.einsum('ij,ij->j',np.matmul(R,np.matmul(A,X))+np.matmul(t,e)-Y,Dir),(m,1))
		err = np.linalg.norm(E-E_old)
		E_old = E
	E = Q - np.matmul(R, np.matmul(A,X)) - np.matmul(t,e)
	
	# calculate fiducial registration error
	fre = np.sum(np.linalg.norm(E,ord=2,axis=0,keepdims=True))/X.shape[1]
	return[R,t,A,Q,fre]

def closest_point_and_distance(p, a, b):
	"""returns both the closest point and the distance.
	"""
	s = b - a
	w = p - a
	ps = np.dot(w.T, s)
	if ps <= 0:
		return [a, np.linalg.norm(w)]
	l2 = np.dot(s.T, s)
	if ps >= l2:
		closest = b
	else:
		closest = a + ps / l2 * s
	return [closest, np.linalg.norm(p - closest)]

dim = 3
n = 10


X = np.random.rand( dim, n )

gt_A = np.identity(dim) * np.random.rand(dim,1)

gt_R=rotation_matrix(pitch,roll, yaw)

gt_t = np.random.rand(3,1) * 10
print('Ground truth rotation:\n', gt_R, '\nGround truth translation\n', gt_t, '\nGround truth scaling\n', gt_A)

e = np.ones((1,n))
Y = np.matmul( np.matmul(gt_R, gt_A), X ) + np.matmul(gt_t,e)

#print(X)
#print(Y)
[R,t,A] = AOPA_Major(X, Y, 1e-9)
print('Computed rotation:\n', R, '\nComputed translation\n', t, '\nComputed scaling\n',A)

test=np.c_[np.array([0.51346663,0.59783665,0.26221566]),np.tile(np.array([0.51346663,0.59783665,0.26221566]),(3,3))]

tm,Q,fre=p2l(X,Y,D,.0005)

frameRotation = slicer.mrmlScene.AddNode(slicer.vtkMRMLLinearTransformNode())
frameRotation.SetName('frame_rotation')
frameRotation.SetMatrixTransformFromParent(tm)
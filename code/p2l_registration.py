#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 08:11:06 2022

@author: greydon
"""

import numpy as np
import math
import pandas as pd

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

def prep_p2l(X,Y):
	R = np.eye(3)
	T = np.zeros((3, 1))
	# Arun equation 4
	p = np.ndarray.mean(X, 0)
	# Arun equation 6
	p_prime = np.ndarray.mean(Y, 0)
	# Arun equation 7
	q = X - p
	# Arun equation 8
	q_prime = Y - p_prime
	# Arun equation 11
	H = np.matmul(q.transpose(), q_prime)
	svd = np.linalg.svd(H)
	VU = np.matmul(svd[2].transpose(), svd[0])
	detVU = np.linalg.det(VU)
	diag = np.eye(3, 3)
	diag[2][2] = detVU
	R = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
	tmp = p_prime.transpose() - np.matmul(R, p.transpose())
	X=X.T
	Y=Y.T
	T[0][0] = tmp[0]
	T[1][0] = tmp[1]
	T[2][0] = tmp[2]
	return X,Y,T


ifile=r'/media/veracrypt6/projects/stealthMRI/derivatives/trajectoryGuide/derivatives/frame/sub-P150_space-leksellg_desc-centroids_fids.tsv'
ifile2=r'/home/greydon/Downloads/sub-P150_space-leksellg_desc-fiducials_fids.fcsv'

df_tmp = pd.read_csv( ifile, header=0,sep='\t')
Y = df_tmp[['ideal_x','ideal_y','ideal_z']].to_numpy()

df_tmp2 = pd.read_csv( ifile2, header=2,sep=',')
X = df_tmp2[['x','y','z']].to_numpy()

R,t,A=AOPA_Major(X.T,Y.T,.0005)
mat = np.dot(R, A)

X.shape,Y.shape,D.shape
R,t,A,Q,fre=p2l(X, Y, D, .0005)

R=R.T
t=t.T

lps2ras = np.diag([-1, -1, 1])
data = np.eye(4)
data[0:3, 3] = t.T
data[:3, :3] = mat

lps2ras=np.diag([-1, -1, 1, 1])
ras2lps=np.diag([-1, -1, 1, 1])
transform_lps=np.dot(data,lps2ras)

transform_matrix = vtk.vtkMatrix4x4()
dimensions = len(transform_lps) - 1
for row in range(dimensions + 1):
	for col in range(dimensions + 1):
		transform_matrix.SetElement(row, col, transform_lps[(row, col)])


inputTransform = slicer.mrmlScene.AddNode(slicer.vtkMRMLLinearTransformNode())
inputTransform.SetName('p2l')
inputTransform.SetMatrixTransformFromParent(transform_matrix)



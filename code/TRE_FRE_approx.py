#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:33:59 2022

@author: greydon
"""
import itertools


def TRE_FRE_approx(X0,W, Cov_FLE, r0):
	"""
	Calculates root-mean-squares of TRE and FRE
	and covariance matrices of TRE and FRE for
	fiducials X0, weightings W, FLE covariance
	matrices Cov_FLE and target r0. X0 is a 3-by-N
	matrix with a fiducial point in each column,
	W is 3-by-3-by-N with a weighting
	matrix for one fiducial on each page. rO is 
	the 3-by-1 target. Note that FRE is the
	weighted fiducial registration error.
	"""

N = X0.shape

Wl = W[:,0,:]
W2 = W[:,1,:]
W3 = W[:,2,:]

Xl = XO[0,:]
X2 = XO[1,:]
X3 = XO[2,:]

Xreshaped = np.r_[np.tile(X1,(3,1)),np.tile (X2,(3,1)), np.tile(X3,(3,1))].reshape(W.shape)

Xl = Xreshaped[:,0,:]
X2 = Xreshaped[:,1,:]
X3 = Xreshaped[:,2,:]

C =[(-W2*X3 + W3*X2), (+W1*X3 - W3*X1), (-W1*X2 + W2*X1), W1, W2, W3]

list(itertools.permutations([1, 2, 3]))

Cc = permute(C, [1,3,2]),[],6).reshape(6)
	
U,S,V=np.lingalg.svd(C)
 

temp = num2cell(W,[1 2])
Wlarge=blkdiag(temp{:})

temp = num2cell (Cov_FLE, [1 2])
Cov_FLElarge = blkdiag(temp{:})

M =U.T* Wlarge * Cov_FLElarge *Wlarge.T *U
% FRE
U(:,1:6)= 0
Cov_FRE = U*M*U.T
RMS_FRE = sqrt(trace(Cov_FRE))

# TRE
Dleft = np.r_[np.c_[0,r0(3),-r0(2],np.c_[-r0(3),0,r0(1)], np.c_[r0(2),-r0(1),0]]
D = np.c_[Dleft, np.eye(3)]
 
DV = D * V3
S1 = np.c_[np.linalg.inv(S[:6, :6]), np.zeros((6, 3*N-6)) ]
Cov_TRE = DV*S1*M*S1'*DV';
RMS_TRE = sqrt (trace (Cov_TRE) )
 
return [RMS_TRE,RMS_FRE,Cov_TRE,Cov_FRE]
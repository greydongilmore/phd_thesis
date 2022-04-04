#  -*- coding: utf-8 -*-

"""Functions for point based registration using Orthogonal Procrustes."""

import numpy as np

# pylint: disable=invalid-name, line-too-long

def expected_absolute_value(std_devs):
	"""
	Returns the expected absolute value of a normal
	distribution with mean 0 and standard deviations std_dev
	"""

	onedsd = np.linalg.norm(std_devs)
	variance = onedsd * onedsd
	return variance

def distance_from_line(p_1, p_2, p_3):
	"""
	Computes distance of a point p_3, from a line defined by p_1 and p_2.
	See `here <https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line>`_.
	:return: euclidean distance
	"""
	# I want notation same as wikipedia page, so disabling warning.
	# pylint: disable=invalid-name
	n = p_2 - p_1
	n = n / np.linalg.norm(n)
	a_minus_p = p_1 - p_3
	vector_to_line = a_minus_p - (np.dot(a_minus_p, n) * n)
	distance = np.linalg.norm(vector_to_line)
	return distance

def validate_procrustes_inputs(fixed, moving):
	"""
	Validates the fixed and moving set of points

	1. fixed and moving must be numpy array
	2. fixed and moving should have 3 columns
	3. fixed and moving should have at least 3 rows
	4. fixed and moving should have the same number of rows

	:param fixed: point set, N x 3 ndarray
	:param moving: point set, N x 3 ndarray of corresponding points
	:returns: nothing
	:raises: TypeError, ValueError
	"""
	if not isinstance(fixed, np.ndarray):
		raise TypeError("fixed is not a numpy array'")

	if not isinstance(moving, np.ndarray):
		raise TypeError("moving is not a numpy array")

	if not fixed.shape[1] == 3:  # pylint: disable=literal-comparison
		raise ValueError("fixed should have 3 columns")

	if not moving.shape[1] == 3:  # pylint: disable=literal-comparison
		raise ValueError("moving should have 3 columns")

	if fixed.shape[0] < 3:
		raise ValueError("fixed should have at least 3 points (rows)")

	if moving.shape[0] < 3:
		raise ValueError("moving should have at least 3 points (rows)")

	if not fixed.shape[0] == moving.shape[0]:
		raise ValueError("fixed and moving should have "
						 + "the same number of points (rows)")


def compute_fre(fixed, moving, rotation, translation):
	"""
	Computes the Fiducial Registration Error, equal
	to the root mean squared error between corresponding fiducials.
	:param fixed: point set, N x 3 ndarray
	:param moving: point set, N x 3 ndarray of corresponding points
	:param rotation: 3 x 3 ndarray
	:param translation: 3 x 1 ndarray
	:returns: Fiducial Registration Error (FRE)
	"""
	# pylint: disable=assignment-from-no-return
	#validate_procrustes_inputs(fixed, moving)
	transformed_moving = np.matmul(rotation, moving.transpose()) + translation
	squared_error_elementwise = np.square(fixed, transformed_moving.transpose())
	square_distance_error = np.sum(squared_error_elementwise, 1)
	sum_squared_error = np.sum(square_distance_error, 0)
	mean_squared_error = sum_squared_error / fixed.shape[0]
	fre = np.sqrt(mean_squared_error)
	return fre, mean_squared_error, squared_error_elementwise


def compute_tre_from_fle(fiducials, mean_fle_squared, target_point):
	"""
	Computes an estimation of TRE from FLE and a list of fiducial locations.

	See:
	`Fitzpatrick (1998), equation 46 <http://dx.doi.org/10.1109/42.736021>`_.

	:param fiducials: Nx3 ndarray of fiducial points
	:param mean_fle_squared: expected (mean) FLE squared
	:param target_point: a point for which to compute TRE.
	:return: mean TRE squared
	"""
	# pylint: disable=literal-comparison
	if not isinstance(fiducials, np.ndarray):
		raise TypeError("fiducials is not a numpy array'")
	if not fiducials.shape[1] == 3:
		raise ValueError("fiducials should have 3 columns")
	if fiducials.shape[0] < 3:
		raise ValueError("fiducials should have at least 3 rows")
	if not isinstance(target_point, np.ndarray):
		raise TypeError("target_point is not a numpy array'")
	if not target_point.shape[1] == 3:
		raise ValueError("target_point should have 3 columns")
	if not target_point.shape[0] == 1:
		raise ValueError("target_point should have 1 row")

	number_of_fiducials = fiducials.shape[0]
	centroid = np.mean(fiducials, axis=0)
	covariance = np.cov(fiducials.T)
	assert covariance.shape[0] == 3
	assert covariance.shape[1] == 3
	_, eigen_vectors_matrix = np.linalg.eig(covariance)

	f_array = np.zeros(3)
	for axis_index in range(3):
		sum_f_k_squared = 0
		for fiducial_index in range(fiducials.shape[0]):
			f_k = distance_from_line(centroid,
										eigen_vectors_matrix[axis_index],
										fiducials[fiducial_index])
			sum_f_k_squared = sum_f_k_squared + f_k * f_k
		f_k_rms = np.sqrt(sum_f_k_squared / number_of_fiducials)
		f_array[axis_index] = f_k_rms

	inner_sum = 0
	for axis_index in range(3):
		d_k = distance_from_line(centroid,
									eigen_vectors_matrix[axis_index],
									target_point)
		inner_sum = inner_sum + (d_k * d_k / (f_array[axis_index] *
											  f_array[axis_index]))

	mean_tre_squared = (mean_fle_squared / fiducials.shape[0]) * \
					   (1 + (1./3.) * inner_sum)
	return mean_tre_squared


def compute_fre_from_fle(fiducials, mean_fle_squared):
	"""
	Computes an estimation of FRE from FLE and a list of fiducial locations.

	See:
	`Fitzpatrick (1998), equation 10 <http://dx.doi.org/10.1109/42.736021>`_.

	:param fiducials: Nx3 ndarray of fiducial points
	:param mean_fle_squared: expected (mean) FLE squared
	:return: mean FRE squared
	"""
	# pylint: disable=literal-comparison
	if not isinstance(fiducials, np.ndarray):
		raise TypeError("fiducials is not a numpy array'")
	if not fiducials.shape[1] == 3:
		raise ValueError("fiducials should have 3 columns")
	if fiducials.shape[0] < 3:
		raise ValueError("fiducials should have at least 3 rows")
	number_of_fiducials = fiducials.shape[0]
	fre_sq = (1 - (2.0 / number_of_fiducials)) * mean_fle_squared
	return fre_sq


def orthogonal_procrustes(fixed, moving):
	"""
	Implements point based registration via the Orthogonal Procrustes method.

	Based on Arun's method:

	  Least-Squares Fitting of two, 3-D Point Sets, Arun, 1987,
	  `10.1109/TPAMI.1987.4767965 <http://dx.doi.org/10.1109/TPAMI.1987.4767965>`_.

	Also see `this <http://eecs.vanderbilt.edu/people/mikefitzpatrick/papers/2009_Medim_Fitzpatrick_TRE_FRE_uncorrelated_as_published.pdf>`_
	and `this <http://tango.andrew.cmu.edu/~gustavor/42431-intro-bioimaging/readings/ch8.pdf>`_.

	:param fixed: point set, N x 3 ndarray
	:param moving: point set, N x 3 ndarray of corresponding points
	:returns: 3x3 rotation ndarray, 3x1 translation ndarray, FRE
	:raises: ValueError
	"""
	# This is what we are calculating
	R = np.eye(3)
	T = np.zeros((3, 1))
	# Arun equation 4
	p = np.ndarray.mean(moving, 0)
	# Arun equation 6
	p_prime = np.ndarray.mean(fixed, 0)
	# Arun equation 7
	q = moving - p
	# Arun equation 8
	q_prime = fixed - p_prime
	# Arun equation 11
	H = np.matmul(q.transpose(), q_prime)
	# Arun equation 12
	# Note: numpy factors h = u * np.diag(s) * v
	svd = np.linalg.svd(H)
	# Replace Arun Equation 13 with Fitzpatrick, chapter 8, page 470,
	# to avoid reflections, see issue #19
	VU = np.matmul(svd[2].transpose(), svd[0])
	detVU = np.linalg.det(VU)
	diag = np.eye(3, 3)
	diag[2][2] = detVU
	X = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
	# Arun step 5, after equation 13.
	det_X = np.linalg.det(X)
	if det_X < 0 and np.all(np.flip(np.isclose(svd[1], np.zeros((3, 1))))):
		# Don't yet know how to generate test data.
		# If you hit this line, please report it, and save your data.
		raise ValueError("Registration fails as determinant < 0"
						 " and no singular values are close enough to zero")
	if det_X < 0 and np.any(np.isclose(svd[1], np.zeros((3, 1)))):
		# Implement 2a in section VI in Arun paper.
		v_prime = svd[2].transpose()
		v_prime[0][2] *= -1
		v_prime[1][2] *= -1
		v_prime[2][2] *= -1
		X = np.matmul(v_prime, svd[0].transpose())
	# Compute output
	R = X
	tmp = p_prime.transpose() - np.matmul(R, p.transpose())
	T[0][0] = tmp[0]
	T[1][0] = tmp[1]
	T[2][0] = tmp[2]
	fre, mean_sq, sq_error = compute_fre(fixed, moving, R, T)
	print(fre)
	return R, T, fre

def _fitzpatricks_X(svd):
	"""This is from Fitzpatrick, chapter 8, page 470.
	   it's used in preference to Arun's equation 13,
	   X = np.matmul(svd[2].transpose(), svd[0].transpose())
	   to avoid reflections.
	"""
	VU = np.matmul(svd[2].transpose(), svd[0])
	detVU = np.linalg.det(VU)
	diag = np.eye(3, 3)
	diag[2][2] = detVU
	X = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
	return X

#%%
import math
import pandas as pd


ifile=r'/media/veracrypt6/projects/stealthMRI/derivatives/trajectoryGuide/derivatives/frame/sub-P150_space-leksellg_desc-centroids_fids.tsv'
ifile2=r'/home/greydon/Downloads/sub-P150_space-leksellg_desc-fiducials_fids.fcsv'
df_tmp = pd.read_csv( ifile, header=0,sep='\t')
fixed_fids = df_tmp[['ideal_x','ideal_y','ideal_z']].to_numpy()

df_tmp2 = pd.read_csv( ifile, header=0,sep='\t')
moving_fids = df_tmp2[['x','y','z']].to_numpy()


fle_sd = np.random.uniform(low=0.5, high=1.0)
#change fle_ratio if you want anisotropic fle
fle_ratio = np.array([.625, .625, 1.25], dtype=np.float64)
anis_scale = math.sqrt(3.0 / (np.linalg.norm(fle_ratio) ** 2))
fixed_fle = fle_ratio * fle_sd * anis_scale
moving_fle = np.array([0., 0., 0.], dtype=np.float64)
fixed_fle_eavs = expected_absolute_value(fixed_fle)
moving_fle_eavs = expected_absolute_value(moving_fle)



rotation, translation, fre = orthogonal_procrustes(fixed_fids, moving_fids)

expected_tre_squared = compute_tre_from_fle(moving_fids[:, 0:3], fixed_fle_eavs, fixed_fids[1, 0:3].reshape(-1,1).T)
expected_fre_sq = compute_fre_from_fle(moving_fids[:, 0:3], fixed_fle_eavs)
mean_fle = math.sqrt(fre)
mean_fre = math.sqrt(expected_fre_sq)

transformed_target = np.matmul(rotation,fixed_fids.transpose()) + translation
actual_tre = np.linalg.norm(transformed_target - fixed_fids[:, 0:3].transpose())


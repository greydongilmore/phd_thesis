#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 05:06:14 2021

@author: greydon
"""

import os
from skimage import morphology
from skimage import measure,segmentation
from skimage import feature
from scipy import ndimage
import nibabel as nb
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from skimage.color import rgb2gray
from scipy import ndimage
import pandas as pd
import subprocess
import vtk
from nilearn import plotting,image
from sklearn.decomposition import PCA
from nilearn.datasets import load_mni152_template
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import median_filter
import cv2
import csv
import glob
import re

os.chdir('/home/greydon/Documents/GitHub/sandbox')

from frameDetection import frameDetection

def sorted_nicely(data, reverse = False):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	
	return sorted(data, key = alphanum_key, reverse=reverse)

frame_settings_leksell={
	'system': 'leksellg',
	'min_threshold': 450,
	'max_threshold': 'n/a',
	'n_markers': 9,
	'n_components': 3,
	'min_size': 10,
	'max_size': 6500,
	'intensity_weight':True,
	'fid_lambda':80,
	'fid_ratio': 2,
	'labels':{
		1:9,2:8,3:7,4:6,5:5,6:4,7:1,8:2,9:3
	},
	'sort_idx':{
		0:[1,2,3,7,8,9],
		1:[4,5,6]
	},
	'localizer_axis':{
		'AP': [[3,2,1],[7,8,9]],
		'ML':[[4,5,6]]
	},
	'localizer_labels':{
		1:'G',2:'H',3:'I',4:'D',5:'E',6:'F',7:'A',8:'B',9:'C'
	},
	'localizer_bar_radius':4,
	'frame_mid_bars':{
		'bar_B':{
			'bot':'bar_C_bot',
			'top':'bar_A_top'
		},
		'bar_E':{
			'bot':'bar_F_bot',
			'top':'bar_D_top'
		},
		'bar_H':{
			'bot':'bar_G_bot',
			'top':'bar_I_top'
		}
	}
}

frame_settings_brw={
	'system': 'brw',
	'min_threshold': 200,
	'max_threshold': 460,
	'n_markers': 9,
	'n_components': 9,
	'min_size': 1000,
	'max_size': 25000,
	'intensity_weight':True,
	'fid_lambda':25,
	'fid_ratio': 10,
	'labels':{
		1:1, 2:2,  3:8,  4:5,  5:6,  6:3,  7:9,  8:7,  9:4
	},
	'sort_idx':{
		0:[1,2,3,4,5,6],
		1:[7,8,9]
	},
	'localizer_axis':{
		'AP': [[3,2,1],[6,5,4]],
		'ML':[[9,8,7]]
	},
	'localizer_labels':{
		1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I'
	},
	'localizer_bar_radius':1,
	'frame_mid_bars':{
		'bar_B':{
			'bot':'bar_A_bot',
			'top':'bar_C_top'
		},
		'bar_E':{
			'bot':'bar_D_bot',
			'top':'bar_F_top'
		},
		'bar_H':{
			'bot':'bar_G_bot',
			'top':'bar_I_top'
		}
	}
}


# 	'Rigidbody (6 DOF)':0
# 	'Similarity (7 DOF)':1
# 	'Affine (9 DOF)':2


icp_settings={
	'parameters':{
		'transformType': 0,
		'numIterations': 100,
		'numLandmarks': 450,
		'matchCentroids': False,
		'reverseSourceTar': False,
		'distanceMetric': 'rms',
		'maximum_mean_distance':0.001
	}
}


#%%

import itertools

frame_settings=frame_settings_leksell
frame_settings['settings']=icp_settings

input_dir=r'/media/veracrypt6/projects/stealthMRI/imaging/clinical/deriv/validation/scenes'
outFolder=r'/media/veracrypt6/projects/stealthMRI/imaging/clinical/deriv/validation'


parameters={
	'matchCentroids': [False,True],
	'numLandmarks':[200,250,300,350,400,450,500,550,600],
	'maximum_mean_distance':[0.01,0.001,0.0001,0.00001,0.000001]
	}

parameters={
	'min_threshold':[200,250,300,350,400,450,500,550,600,650,700]
	}


parameters={
	'min_threshold':np.arange(100,320,20)
	}

#parameters={'none':[None]}


varNames = sorted(parameters)
param_matrix = [dict(zip(varNames, prod)) for prod in itertools.product(*(parameters[varName] for varName in varNames))]

outFolder_thres = os.path.join(outFolder,"registration_error")

if not os.path.exists(outFolder_thres):
	os.makedirs(outFolder_thres)


for ifile in sorted_nicely([x for x in glob.glob(input_dir+'/*/*Frame*.nii.gz')]):
	subject = os.path.basename((os.path.dirname(ifile)))
	cnt=1
	for iparam in param_matrix:
		subject = os.path.basename((os.path.dirname(ifile)))
		filen=f"{subject}_space-{frame_settings['system']}_"
		if not list(iparam)[0] =='none':
			for key in list(iparam):
				if key == 'min_threshold':
					frame_settings[key]=iparam[key]
				else:
					frame_settings['settings']['parameters'][key]=iparam[key]
				val = iparam[key]
				if key == 'maximum_mean_distance':
					val='{:f}'.format(val).rstrip('0')
				filen+=f"{key.replace('_','')}-{val}_"
		
		outfile_name_clus = os.path.join(os.path.join(outFolder_thres,subject, 'frame',filen+"desc-clusters_fids.tsv"))
		outfile_name_cen = os.path.join(os.path.join(outFolder_thres,subject, 'frame',filen+"desc-centroids_fids.tsv"))
		
		if not os.path.exists(outfile_name_clus):
			
			if not os.path.exists(os.path.join(outFolder_thres,subject)):
				os.makedirs(os.path.join(outFolder_thres,subject))
			
			frameDetectInstance = frameDetection(ifile, os.path.join(outFolder_thres,subject), frame_settings)
			
			frameFiducialPoints = {}
			frameFiducialPoints['label'] = [int(x) for x in frameDetectInstance.final_location_clusters[:,3]]
			frameFiducialPoints['x'] = [format (x, '.3f') for x in frameDetectInstance.final_location_clusters[:,0]]
			frameFiducialPoints['y'] = [format (x, '.3f') for x in frameDetectInstance.final_location_clusters[:,1]]
			frameFiducialPoints['z'] = [format (x, '.3f') for x in frameDetectInstance.final_location_clusters[:,2]]
			frameFiducialPoints['intensity'] = [format (x, '.3f') for x in frameDetectInstance.final_location_clusters[:,4]]
			
			if not os.path.exists(os.path.dirname(outfile_name_clus)):
				os.makedirs(os.path.dirname(outfile_name_clus))
			
			with open(outfile_name_clus, 'w') as out_file:
				writer = csv.writer(out_file, delimiter = "\t")
				writer.writerow(frameFiducialPoints.keys())
				writer.writerows(zip(*frameFiducialPoints.values()))
			
			frameFiducialPoints={}
			frameFiducialPoints['label']=[int(x) for x in frameDetectInstance.final_location[:,3]]
			frameFiducialPoints['x']=[format (x, '.3f') for x in frameDetectInstance.sourcePoints[:,0]]
			frameFiducialPoints['y']=[format (x, '.3f') for x in frameDetectInstance.sourcePoints[:,1]]
			frameFiducialPoints['z']=[format (x, '.3f') for x in frameDetectInstance.sourcePoints[:,2]]
			frameFiducialPoints['intensity']=[format (x, '.3f') for x in frameDetectInstance.final_location[:,4]]
			frameFiducialPoints['error']=[format (x, '.3f') for x in frameDetectInstance.pointError]
			frameFiducialPoints['dist_x']=[format (x, '.3f') for x in frameDetectInstance.pointDistanceXYZ[:,0]]
			frameFiducialPoints['dist_y']=[format (x, '.3f') for x in frameDetectInstance.pointDistanceXYZ[:,1]]
			frameFiducialPoints['dist_z']=[format (x, '.3f') for x in frameDetectInstance.pointDistanceXYZ[:,2]]
			frameFiducialPoints['n_cluster']=[int(x) for x in frameDetectInstance.final_location[:,5]]
			frameFiducialPoints['ideal_x']=[format (x, '.3f') for x in frameDetectInstance.idealPoints[:,0]]
			frameFiducialPoints['ideal_y']=[format (x, '.3f') for x in frameDetectInstance.idealPoints[:,1]]
			frameFiducialPoints['ideal_z']=[format (x, '.3f') for x in frameDetectInstance.idealPoints[:,2]]
			
			with open(outfile_name_cen, 'w') as out_file:
				writer = csv.writer(out_file, delimiter = "\t")
				writer.writerow(frameFiducialPoints.keys())
				writer.writerows(zip(*frameFiducialPoints.values()))
			
			out_msg=f"Finished {subject} parameters {cnt} of {len(param_matrix)}: "
			if not list(iparam)[0] =='none':
				for key in list(iparam):
					out_msg+=f"{key} {iparam[key]} "
			
			print(out_msg)
			cnt +=1
		

















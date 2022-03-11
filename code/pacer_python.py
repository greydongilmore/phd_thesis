#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:37:11 2021

@author: greydon
"""
import os
import argparse
import numpy as np
import numpy_groupies as npg
import pandas as pd
from itertools import product,combinations
from scipy import integrate, signal, optimize,ndimage
from scipy.io import loadmat
from scipy.interpolate import griddata,LinearNDInterpolator
from scipy.signal import filtfilt, medfilt, find_peaks
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import median_filter
from skimage import morphology,measure
import xml.etree.ElementTree as ET
import nibabel as nb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def polyArcLength3(polyCoeff, lowerLimits, upperLimits):
	
	regX = polyCoeff[:,0]
	regY = polyCoeff[:,1]
	regZ = polyCoeff[:,2]
	
	x_d = np.polyder(regX)
	y_d = np.polyder(regY)
	z_d = np.polyder(regZ)
	
	arcLength = []
	returnInt=False
	if isinstance(lowerLimits,int):
		returnInt=True
		lowerLimits=[lowerLimits]
		upperLimits=[upperLimits]
	
	for i in range(len(lowerLimits)):
		f = lambda t: np.sqrt(np.polyval(x_d, t)**2 + np.polyval(y_d, t)**2 + np.polyval(z_d, t)**2)
		result = integrate.quad(f,lowerLimits[i],upperLimits[i])[0]
		arcLength.append(result)
	
	if returnInt:
		arcLength=arcLength[0]
	
	return arcLength

def fitParamPolyToSkeleton(skeleton, degree):
	
	diffVecs = np.diff(skeleton, axis=0)
	apprTotalLengthMm = 0
	deltas=[]
	cumLengthMm=[]
	for idiff in diffVecs:
		deltas.append(np.linalg.norm(idiff))
		cumLengthMm.append(sum(deltas[0:]))
		apprTotalLengthMm+=deltas[-1]
	
	avgTperMm = 1 / apprTotalLengthMm
	t = cumLengthMm/apprTotalLengthMm
	t = np.r_[0,t]

	T = np.ones((len(t),1))
	for d in range(1,degree+1):
		T = np.c_[t**d, T]
	
	T = T.T
	
	r3polynomial=np.linalg.pinv(T).T.dot(skeleton)
	
	fittingErrs = np.sqrt(sum((r3polynomial.T @ T - skeleton.T)**2))
	meanFittingError = np.mean(fittingErrs)
	stdFittingError = np.std(fittingErrs)
	maxFittingError = max(fittingErrs)
	
	print(f"Max off-model: {maxFittingError}\nMean off-model:  {meanFittingError}\n")
	
	if maxFittingError > 0.3 and maxFittingError > (meanFittingError + 3*stdFittingError):
		print('Check for outliers / make sure that the polynomial degree choosen is appropriate.\n In most cases this should be fine.\n')
	
	return r3polynomial, avgTperMm

# elecPointCloudMm=iElec['pointCloudWorld']

def electrodePointCloudModelEstimate(elecPointCloudMm, args):
	revDir = False
	INTERNAL_DEGREE = 8
	USE_REF_IMAGE_WEIGHTING=False
	
	if args.reverseDir:
		revDir = True
	
	
	#Axial centers based skeletonization
	zPlanes = np.unique(elecPointCloudMm[:,2])
	tol = 0
	if not len(zPlanes) < len(elecPointCloudMm):
		tol = 0.1
		i = np.argsort(elecPointCloudMm[:,2].flat)
		d = np.append(True, np.diff(elecPointCloudMm[:,2].flat[i]))
		zPlanes = elecPointCloudMm[:,2].flat[i[d>tol / max(abs(elecPointCloudMm[:,2]))]]
	
	skeleton = []
	sumInPlane = []
	
	for iPlane in zPlanes:
		inPlanePoints = elecPointCloudMm[abs(elecPointCloudMm[:,2] -iPlane) <= tol ,:]
		if len(inPlanePoints) > 1:
# 			if USE_REF_IMAGE_WEIGHTING:
# 				inPlaneIntensities = single(pixelValues(abs(elecPointCloudMm(:,3) - iPlane) <= tol)) # pixelValues MUST be same order than elecPointCloudMm!
# 				skeleton[-1,:] = inPlanePoints.T * inPlaneIntensities / sum(inPlaneIntensities).T
# 				sumInPlane[-1] = sum(inPlaneIntensities)
# 			else
			aver=inPlanePoints.mean(axis=0)
			skeleton.append(aver.tolist())
		
	skeleton = np.array(skeleton)
	
# 	# Filter Skeleton for valid Points
# 	filt = sumInPlane < (np.median(sumInPlane) / 1.5)
# 	if sum(filt) > 0:
# 		skeleton = skeleton(~filt,:)
	
	# Approximate parameterized polynomial  ([x y z] = f(t))
	if len(skeleton) < INTERNAL_DEGREE + 1:
		INTERNAL_DEGREE = len(skeleton) - 1
	
	if revDir:
		r3polynomial, tPerMm = fitParamPolyToSkeleton(np.flipud(skeleton), INTERNAL_DEGREE)
	else:
		r3polynomial, tPerMm = fitParamPolyToSkeleton(skeleton, INTERNAL_DEGREE)
	
	totalLengthMm = polyArcLength3(r3polynomial, 0, 1)
	
	return r3polynomial, tPerMm, skeleton, totalLengthMm

# r3polyToUse=initialR3polynomial

def oor(r3polyToUse, STEP_SIZE, XGrid, YGrid, interpolationF):
	
	SND_THRESH=1500

	arcLength = polyArcLength3(r3polyToUse, 0, 1)
	oneMmEqivStep = 1 / arcLength
	lookahead = 3 * oneMmEqivStep
	
	regX = r3polyToUse[:,0]
	regY = r3polyToUse[:,1]
	regZ = r3polyToUse[:,2]
	orthogonalSamplePoints = []
	improvedSkeleton = []
	avgIntensity = []
	medIntensity = []
	sumIntensity = []
	
	orthSamplePointsVol=[]
	orthIntensVol=[]
	evalAtT_final=[]
	
	evalAtT=np.arange(-lookahead, 1, STEP_SIZE)
	for evalAt in evalAtT:
		x_d = np.polyval(np.polyder((regX)), evalAt)
		y_d = np.polyval(np.polyder((regY)), evalAt)
		z_d = np.polyval(np.polyder((regZ)), evalAt)
		
		currentPoint = np.polyval(r3polyToUse, evalAt).T
		direction =np.array([x_d, y_d, z_d])
		directionNorm = direction / np.linalg.norm(direction)
		
		ortho1 = np.cross(directionNorm.T, np.array([0, 1, 0]).T)
		ortho1 = ortho1 / np.linalg.norm(ortho1)
		ortho2 = np.cross(directionNorm.T, ortho1)
		ortho2 = ortho2 / np.linalg.norm(ortho2)
		
		orthogonalSamplePoints = np.add(currentPoint, (ortho1 * np.tile(XGrid.ravel(), (3,1)).T) + (ortho2 * np.tile(YGrid.ravel(), (3,1)).T))
		orthSamplePointsVol.append(orthogonalSamplePoints)
		
		intensities = interpolationF(orthogonalSamplePoints)
		intensitiesNanZero = intensities.copy()
		intensitiesNanZero[np.isnan(intensitiesNanZero)] = 0
		intensitiesNanZero[intensities<SND_THRESH] = 0
		
		with np.errstate(divide='ignore', invalid='ignore'):
			skelPoint = orthogonalSamplePoints.T.dot(intensitiesNanZero/sum(intensitiesNanZero))
		
		if not any(np.isnan(skelPoint.ravel())):
			avgIntensity.append(np.nanmean(intensities))
			sumIntensity.append(np.nansum(intensitiesNanZero))
			medIntensity.append(np.nanmedian(intensities))
			
			improvedSkeleton.append(skelPoint)
			intensityMap = np.reshape(intensitiesNanZero, np.shape(XGrid))
		
			orthIntensVol.append(intensityMap)
			
			evalAtT_final.append(evalAt)
			
	evalAtT_final=np.array(evalAtT_final)
	improvedSkeleton=np.stack(improvedSkeleton)
	orthSamplePointsVol=np.stack(orthSamplePointsVol)
	
	lowerLimits = np.zeros((len(evalAtT_final)))
	upperLimits = evalAtT_final
	lowerLimits[upperLimits < 0] = upperLimits[upperLimits < 0]
	upperLimits[upperLimits < 0] = 0
	skelScaleMm = np.array(polyArcLength3(r3polyToUse, lowerLimits, upperLimits))
	skelScaleMm[lowerLimits < 0] = -1*skelScaleMm[lowerLimits < 0]
	
	return improvedSkeleton, medIntensity,orthIntensVol, orthSamplePointsVol, skelScaleMm

def getIntensityPeaks(filteredIntensity, skelScaleMm, filterIdxs):
	
	pos = find_peaks(filteredIntensity[filterIdxs], height=1.1 * np.nanmean(filteredIntensity), prominence=0.01 * np.nanmean(filteredIntensity), distance=1.4)
	
	try:
		threshold = min(pos[1]['peak_heights'][:4]) - (min(pos[1]['prominences'][:4]) / 4)
		threshIntensityProfile = np.minimum(filteredIntensity, threshold)
		contactSampleLabels = measure.label(~(threshIntensityProfile[filterIdxs] < threshold))
		values = npg.aggregate(contactSampleLabels, skelScaleMm[filterIdxs], func='sum', fill_value=0)
		counts = npg.aggregate(contactSampleLabels, 1, func='sum', fill_value=0)
		peakWaveCenters = values[1:5]/counts[1:5]
	except:
		print('peakWaveCenter detection failed. Returing peaksLocs in peakWaveCenters.')
		peakWaveCenters = skelScaleMm[filterIdxs][pos[0]]
		threshIntensityProfile = filteredIntensity
		threshold = np.nan
	
	thresholdArea = np.mean(filteredIntensity[filterIdxs])
	threshIntensityProfileArea = np.minimum(filteredIntensity, thresholdArea)
	contactSampleLabels, n_labels = measure.label(~(threshIntensityProfileArea[filterIdxs] < thresholdArea), background=0, return_num=True)
	values = npg.aggregate(contactSampleLabels, skelScaleMm[filterIdxs], func='sum', fill_value=0)
	counts = npg.aggregate(contactSampleLabels, 1, func='sum', fill_value=0)
	# 0: "zero label", 1: contact region, 2: X-Ray marker
	contactAreaCenter = values[1]/counts[1]
	
	idxs = np.where(contactSampleLabels+1==2)[0]
	contactAreaWidth =  abs(skelScaleMm[idxs[0]]-skelScaleMm[idxs[-1]])
	if max(contactSampleLabels) > 1:
		print('Multiple metal areas found along electrode. Is this an electrode type with an addtional X-Ray marker?')
		xrayMarkerAreaCenter = values[2]/counts[2]
		idxs = np.where(contactSampleLabels+1==3)[0]
		xrayMarkerAreaWidth =  abs(skelScaleMm[idxs[0]]-skelScaleMm[idxs[-1]])
	
	peaks={
		'peakValues':pos[1]['peak_heights'],
		'peakLocs':skelScaleMm[filterIdxs][pos[0]],
		'pkPromineces':pos[1]['prominences'],
		'peakValues':pos[1]['peak_heights'],
		'peakWaveCenters':peakWaveCenters,
		'threshIntensityProfile':threshIntensityProfile,
		'threshold':threshold,
		'contactAreaCenter':contactAreaCenter,
		'contactAreaWidth':contactAreaWidth,
		'xrayMarkerAreaCenter':xrayMarkerAreaCenter,
		'xrayMarkerAreaWidth':xrayMarkerAreaWidth
	}
		
	return peaks

def determineElectrodeType(peakDistances, args):
	electrodeGeometries = loadmat(args.electrodeGeometries)
	
	distances = []
	rms =[]
	for imodel in electrodeGeometries['electrodeGeometries'][0]:
		try:
			distances.append(np.linalg.norm(np.diff(peakDistances).reshape(-1,1) - imodel['diffsMm']))
			rms.append(np.sqrt(np.mean((np.diff(peakDistances).reshape(-1,1) - imodel['diffsMm'])**2)))
		except:
			distances.append(np.inf)
			rms.append(np.inf)
	
	if np.all((np.isinf(distances))):
		print('determineElectrodeType: Could NOT detect electrode type! Electrode contact detection might by flawed. To low image resolution (to large slice thickness)!? Set electrode type manually if you want to continue with this data')
		elecStruct = electrodeGeometries['electrodeGeometries'][0][-1]
		rms=np.inf
	else:
		idx = np.argmin(distances)
		d=distances[idx]
		rms = rms[idx]
		elecStruct = electrodeGeometries['electrodeGeometries'][0][idx]
	
	return elecStruct,rms

def invPolyArcLength3(polyCoeff, arcLength):
	t=[]
	if isinstance(arcLength,float):
		arcLength=[arcLength]
	
	if isinstance(arcLength,int):
		arcLength=[arcLength]
	
	for iarc in arcLength:
		if iarc<0:
			print('invPolyArcLength3: given arcLength is negative! Forcing t=0. This is wrong but might be approximatly okay for the use case! Check carefully!')
		else:
			func = lambda b: abs(iarc - polyArcLength3(polyCoeff,0,b))
			t.append(optimize.fmin(func=func, x0=[0]))
	return t

# pointCloudWorld=iElec['pointCloudWorld']
# voxelValues=iElec['pixelValues']

def refitElec(initialR3polynomial, pointCloudWorld, voxelValues, args):
	
	XY_RESOLUTION = 0.1
	Z_RESOLUTION = 0.025
	LIMIT_CONTACT_SEARCH_MM = 20
	
	FINAL_DEGREE = args.finalDegree
	DISPLAY_PROFILES = args.displayProfiles
	DISPLAY_MPR = args.displayMPR

	if max(args.voxsize) > 1:
		args.contactDetectionMethod = 'contactAreaCenter'
	
	interpolationF= LinearNDInterpolator(pointCloudWorld, voxelValues.astype(float))
	
	totalLengthMm  = polyArcLength3(initialR3polynomial, 0, 1)
	
	XGrid,YGrid = np.meshgrid(np.arange(-1.5, 1.5, XY_RESOLUTION), np.arange(-1.5, 1.5, XY_RESOLUTION))
	oneMmEqivStep = 1 / totalLengthMm
	STEP_SIZE = Z_RESOLUTION * oneMmEqivStep
	
	### 2nd Pass
	skeleton2nd, _,_, _, _ = oor(initialR3polynomial, STEP_SIZE, XGrid, YGrid, interpolationF)
	refittedR3Poly2nd = fitParamPolyToSkeleton(skeleton2nd, 8)[0]
	
	## additional call to get 2nd pass refitted Poly values for skelScaleMm and medIntensity
	skeleton3rd, medIntensity, orthIntensVol, _, skelScaleMm = oor(refittedR3Poly2nd, STEP_SIZE, XGrid, YGrid, interpolationF)
	
	print(f'1st Pass Electrode Length within Brain Convex Hull: {np.round(polyArcLength3(initialR3polynomial, 0, 1),3)} mm' )
	print(f'2nd Pass Electrode Length within Brain Convex Hull: {np.round(polyArcLength3(refittedR3Poly2nd, 0, 1),3)} mm' )
	
	filterWidth = int(0.25 / Z_RESOLUTION) + 1
	filteredIntensity = filtfilt(np.ones((filterWidth)), filterWidth, medIntensity)
	filterIdxs = np.where(skelScaleMm <= LIMIT_CONTACT_SEARCH_MM)[0]
	
	peakInfo = getIntensityPeaks(filteredIntensity, skelScaleMm, filterIdxs)
	
	# Decide on contact detection method: TODO refactor
	if args.contactDetectionMethod == 'peakWaveCenters':
		contactPositions =peakInfo['peakWaveCenters']
	else:
		contactPositions = peakInfo['peakLocs']
	
	print(f'refitElec: selected contactDetectionMethod: {args.contactDetectionMethod}')
	try:
		electrodeInfo, dataModelPeakRMS = determineElectrodeType(contactPositions,args)
	except:
		print('Falling back to contactDectionMethod: contactAreaCenter')
		args.contactDetectionMethod = 'contactAreaCenter'
	
	useDetectedContactPositions = 1
	if np.all([len(contactPositions) < 4, args.contactDetectionMethod == 'contactAreaCenter']):
		useDetectedContactPositions = 0
		electrodeGeometries = loadmat(args.electrodeGeometries)
		
		if not args.electrodeType:
			if peakInfo['contactAreaWidth'] < 10.5:
				print('Assuming Medtronic 3389 or Boston Scientific Vercise Directional. Setting 3389.')
				electrodeInfo  = electrodeGeometries['electrodeGeometries'][0][0]
			else:
				print('Assuming Medtronic 3387. Setting 3387.')
				electrodeInfo  = electrodeGeometries['electrodeGeometries'][0][1]
		else:
			print(f'Setting user specified electrode type: {args.electrodeType}')
			idx = [i for i,x in enumerate([x['string'][0] for x in electrodeGeometries['electrodeGeometries'][0]]) if args.electrodeType == x]
			if idx:
				electrodeInfo=electrodeGeometries['electrodeGeometries'][0][idx[0]]
			else:
				print('Unknown electrode type given')
				return
			
		zeroT = invPolyArcLength3(refittedR3Poly2nd, peakInfo['contactAreaCenter']-np.mean(electrodeInfo['ringContactCentersMm'][0][0]))
	else:
		if dataModelPeakRMS > 0.3:
			print('Switching to model based contact positions because of high RMS (Setting useDetectedContactPositions = 0).')
			useDetectedContactPositions = 0
		
		zeroT = invPolyArcLength3(refittedR3Poly2nd, contactPositions[0]-electrodeInfo['zeroToFirstPeakMm'][0][0])[0]
		refittedContactDistances = peakInfo['peakLocs'] - (contactPositions[0]-electrodeInfo['zeroToFirstPeakMm'][0][0])
	
	if FINAL_DEGREE == 1:
		elecEndT = invPolyArcLength3(refittedR3Poly2nd, LIMIT_CONTACT_SEARCH_MM)[0][0].astype(float)
		evalAt=np.linspace(zeroT[0], elecEndT, int(totalLengthMm  / XY_RESOLUTION))
		polySkeleton=np.c_[np.polyval(refittedR3Poly2nd[:,0], evalAt), np.polyval(refittedR3Poly2nd[:,1], evalAt), np.polyval(refittedR3Poly2nd[:,2], evalAt)]
		refittedR3PolyTmp = fitParamPolyToSkeleton(polySkeleton,FINAL_DEGREE)[0]
		evalAt2=np.linspace(0,invPolyArcLength3(refittedR3PolyTmp,totalLengthMm)[0][0],int(totalLengthMm  / XY_RESOLUTION))
		polySkeleton2=np.c_[np.polyval(refittedR3PolyTmp[:,0], evalAt2), np.polyval(refittedR3PolyTmp[:,1], evalAt2), np.polyval(refittedR3PolyTmp[:,2], evalAt2)]
		refittedR3PolyReZeroed = fitParamPolyToSkeleton(polySkeleton2, FINAL_DEGREE)[0]
	else:
		evalAt=np.linspace(zeroT[0], 1, int(totalLengthMm  / XY_RESOLUTION))
		polySkeleton=np.c_[np.polyval(refittedR3Poly2nd[:,0], evalAt), np.polyval(refittedR3Poly2nd[:,1], evalAt), np.polyval(refittedR3Poly2nd[:,2], evalAt)]
		refittedR3PolyReZeroed = fitParamPolyToSkeleton(polySkeleton,FINAL_DEGREE)[0]
	
# 	refitReZeroedElecMod = PolynomialElectrodeModel(refittedR3PolyReZeroed, electrodeInfo)
# 	refitReZeroedElecMod.useDetectedContactPositions = 1
# 	refitReZeroedElecMod.detectedContactPositions = refittedContactDistances[0:electrodeInfo.noRingContacts,:].T
	
	return refittedR3PolyReZeroed, filteredIntensity, skelScaleMm


def largestConnCompSliceWise(img3d):
	
	binaryMask = np.empty(img3d.shape, dtype=bool)
	
	for slice_idx in range(img3d.shape[2]):
		labels, n_labels = measure.label(img3d[:,:,slice_idx], background=0, return_num=True)
		ccProps = measure.regionprops(labels, img3d[:,:,slice_idx])
		idx = np.argsort([prop.area for prop in ccProps])
		
		binarySlice = np.empty(np.shape(img3d[:,:,slice_idx]), dtype=bool)
		if not idx.size==0:
			brainMaskIdxs = ccProps[idx[-1]].coords
			binarySlice[brainMaskIdxs] = True
		
		binaryMask[:,:,slice_idx] = binarySlice
	
	return binaryMask


def flood_fill_hull(image):
	points = np.transpose(np.where(image))
	hull = ConvexHull(points)
	deln = Delaunay(points[hull.vertices]) 
	idx = np.stack(np.indices(image.shape), axis = -1)
	out_idx = np.nonzero(deln.find_simplex(idx) + 1)
	out_img = np.zeros(image.shape)
	out_img[out_idx] = 1
	return out_img, hull

def threshold_image(image, window_center, window_width):
	img_min = window_center - window_width
	img_max = window_center + window_width
	threshold_image = (image > img_min)* (image < img_max)
	return threshold_image

def crop_image(image):
	# Create a mask with the background pixels
	mask = image == 0

	# Find the brain area
	coords = np.array(np.nonzero(~mask))
	top_left = np.min(coords, axis=1)
	bottom_right = np.max(coords, axis=1)
	
	# Remove the background
	croped_image = image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1],:]
	
	return croped_image

def add_pad(image, new_height=512, new_width=512):
	height, width,depth = image.shape

	final_image = np.zeros((new_height, new_width, depth))

	pad_left = int((new_width - width) / 2)
	pad_top = int((new_height - height) / 2)
	
	# Replace the pixels with the image's pixels
	final_image[pad_top:pad_top + height, pad_left:pad_left + width,:] = image
	
	return final_image

def flood_fill_hull(image):
	points = np.transpose(np.where(image))
	hull = ConvexHull(points)
	deln = Delaunay(points[hull.vertices]) 
	idx = np.stack(np.indices(image.shape), axis = -1)
	out_idx = np.nonzero(deln.find_simplex(idx) + 1)
	out_img = np.zeros(image.shape)
	out_img[out_idx] = 1
	return out_img, hull

def extractBrainConvHull(niiCT):
	
	brain_ct_min = 20
	brain_ct_max = 60
	
	print('Applying median filter to raw data...')
	ctIso = niiCT.get_fdata(caching='fill')
	
	ctIso[ctIso < -1024] = -1024
	ctIso[ctIso > 3071] = 3071

	ctIsoMedFilt = median_filter(ctIso, size=(3,3,3))
	
	threshold_image = (ctIsoMedFilt > brain_ct_min)&(ctIsoMedFilt < brain_ct_max)
	
	structEle=morphology.ball(np.ceil(2 / max((niiCT.header["pixdim"])[1:4])))
	morphImg = morphology.binary_opening(threshold_image, structEle)
	
	labels, n_labels = measure.label(morphImg, background=0, return_num=True)
	label_count = np.bincount(labels.ravel().astype(np.int))
	label_count[0] = 0
	
	morphFraction = sum(label_count) / np.size(labels)
	print(f"Morph fraction: {morphFraction}")
	if morphFraction < 0.14 or morphFraction > 0.3:
		print('Uncommon CT data fraction in range (15-60 HU), trying binary close then erode...')
		morphImg = morphology.binary_closing(threshold_image, structEle)
		morphImg = morphology.binary_erosion(morphImg, structEle)
		
		labels, n_labels = measure.label(morphImg, background=0, return_num=True)
		label_count = np.bincount(labels.ravel().astype(np.int))
		label_count[0] = 0
		
		morphFraction = sum(label_count) / np.size(labels)
		print(f"Modified morph fraction: {morphFraction}")
	
	ccProperties = measure.regionprops(labels)
	ccProperties.sort(key=lambda x: x.area, reverse=True)
	areas=np.array([prop.area for prop in ccProperties])
	
	#mask = labels == label_count.argmax()
	
	mask = largestConnCompSliceWise(morphImg)
	
	masked_image = ctIsoMedFilt
	masked_image[~mask] = np.nan
	mask = threshold_image(masked_image,WINDOW_CENTER_CT_BRAIN, WINDOW_WIDTH_CT_BRAIN)

	print('Extracting convex hull brain mask...')
	convHullBrainMask=flood_fill_hull(mask)
	
	print('Running binary erosion on brain convex hull...')
	convHullBrainMask = morphology.binary_erosion(convHullBrainMask[0], structEle)
	convHullBrainMask = morphology.binary_erosion(convHullBrainMask, structEle)
	convHullBrainMask = morphology.binary_erosion(convHullBrainMask, structEle)
	
	return convHullBrainMask, niiCT

def extractElectrodePointclouds(niiCTFile, args):
	
	niiCT = nb.load(niiCTFile)
	
	voxel_dims = (niiCT.header["dim"])[1:4]
	voxsize = (niiCT.header["pixdim"])[1:4]
	
	if args.seegData:
		LAMBDA_1=10
		latent_ratio=5
		METAL_THRESHOLD=1600
	else:
		LAMBDA_1 = 25
		latent_ratio=10
		if niiCT.get_fdata().min() >= 0:
			METAL_THRESHOLD = args.metalThreshold + 1024
		else:
			METAL_THRESHOLD = args.metalThreshold
	
	if args.noMask:
		print('Using NO brain mask as "noMask" parameter was set...')
		brainMask = np.empty(voxel_dims, dtype=bool)
	elif args.brainMask: #TODO:segmentation
		print('Using brain mask provied by parameter "brainMask"...')
		niiBrainMask = nb.load(args[0].brainMask)
		brainMask = niiBrainMask.get_fdata()
	else:
		brainMask, niiCT=extractBrainConvHull(niiCT)
	
	structEle=morphology.ball(np.ceil(3 / max((niiCT.header["pixdim"])[1:4])))
	brainMask = morphology.binary_erosion(brainMask, structEle)
	
	maskedImg = niiCT.get_fdata(caching='fill')
	maskedImg[~brainMask] = np.nan
	threImg = (maskedImg > METAL_THRESHOLD)
	
	if args.display:
		fig, ax = plt.subplots(1, 1)
		tracker = IndexTracker(ax,niiCT.get_fdata(), rotate=True)
		fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
		plt.title('Original Image')
		plt.show()
		
		fig, ax = plt.subplots(1, 1)
		tracker2 = IndexTracker(ax,eroded, rotate=True)
		fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
		plt.title('Brain mask')
		plt.show()
		
		fig, ax = plt.subplots(1, 1)
		tracker3 = IndexTracker(ax,threImg, rotate=True)
		fig.canvas.mpl_connect('scroll_event', tracker3.on_scroll)
		plt.title('Metal threshold mask')
		plt.show()
		
		cropped_image=crop_image(masked_image)
		pad_image=add_pad(cropped_image, new_height=masked_image.shape[0], new_width=masked_image.shape[1])
		
	
	labels, n_labels = measure.label(maskedImg, background=0, return_num=True)
	
	properties = measure.regionprops(labels)
	properties.sort(key=lambda x: x.area, reverse=True)
	areas=np.array([prop.area for prop in properties])
	
	if args.seegData:
		minVoxelNumber = 4
		maxVoxelNumber = 600
	else:
		minVoxelNumber = (1.2 * (1.27/2))**2 * np.pi * 40 / np.prod(voxsize)
		maxVoxelNumber = (4 * (1.27/2))**2 * np.pi * 80 / np.prod(voxsize)
	
	largeComponentsIdx = [int(x) for x in np.where(np.logical_and(areas >= minVoxelNumber, areas <= maxVoxelNumber))[0]]
	
	elecIdxs = []
	for icomp in largeComponentsIdx:
		component=properties[icomp]
		repitition=np.tile(voxsize, (len(component.coords), 1))
		pca = PCA()
		X_pca = pca.fit_transform(component.coords * repitition)
		
		if len(pca.explained_variance_) < 3:
			pass
		
		latent = np.sqrt(pca.explained_variance_) * 2
		lowerAxesLength = sorted(latent[1:3])
		#elecIdxs.append(icomp)
		#print(latent[0], latent[0] / np.mean(latent[1:3]))
		if args.seegData:
			if latent[0] >1 and latent[0] / np.mean(latent[1:3]) >2 and lowerAxesLength[1] / (lowerAxesLength[0] + 0.001) < 8:
				elecIdxs.append(icomp)
		else:
			if latent[0] > LAMBDA_1 and latent[0] / np.mean(latent[1:3]) > latent_ratio and lowerAxesLength[1] / (lowerAxesLength[0] + 0.001) < 8:
				elecIdxs.append(icomp)
	
	if len(elecIdxs) == 0:
		if METAL_THRESHOLD < 3000:
			
			args.metalThreshold=METAL_THRESHOLD * 1.5
			print(f"Retrying extraction with a metal threshold of {METAL_THRESHOLD * 1.5}")
			elecPointCloudsStruct, brainMask = extractElectrodePointclouds(niiCTFile, args2)
	
	XMLDefintionPresent = False
	if args.medtronicXMLPlan !='':
		if os.path.exists(args.medtronicXMLPlan):
			reportedElecs = readMedtronicXMLTrajectory((args.medtronicXMLPlan))
			XMLDefintionPresent = True
			nReportedElecs = length(reportedElecs.trajects)
	
	elecPointCloudsStruct = {}
	for iElec in elecIdxs:
		elec_temp={}
		elec_temp['pixelIdxs']= properties[iElec].coords
		elec_temp['labels']= np.array([labels[x[0],x[1],x[2]] for x in properties[iElec].coords])
		elec_temp['pixelValues'] =np.array([int(niiCT.get_fdata()[x[0],x[1],x[2]]) for x in properties[iElec].coords])
		elec_temp['pointCloudMm'] =(properties[iElec].coords.copy()-1) @ abs(niiCT.affine[0:3,0:3])
		
		voxelIdxList=np.vstack(([properties[iElec].coords.T-1,np.ones((1, np.size(properties[iElec].coords.T,1)))]))
		worldCoordinates=np.dot(niiCT.affine, voxelIdxList)
		elec_temp['pointCloudWorld']=worldCoordinates[0:3,:].T
		
		elecMask = np.empty(np.shape(maskedImg), dtype=bool)
		elecMask[elec_temp['pixelIdxs']] = True
		elec_temp['binaryMaskImage']=elecMask
		
		elecPointCloudsStruct[iElec]=elec_temp
	
	values='pixelIdxs'
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for idx, item in elecPointCloudsStruct.items():
		ax.scatter(item[values][:,0], item[values][:,1], item[values][:,2],s=2)
	ax.set_xlim([0,thresh_img.shape[0]])
	ax.set_ylim([0,thresh_img.shape[1]])
	ax.set_zlim([0,thresh_img.shape[2]])
# 	
# 	final_coors=[]
# 	for idx, item in elecPointCloudsStruct.items():
# 		final_coors.append(item[values])
# 	
# 	final_coors=np.vstack(final_coors)
# 	bandwidth = estimate_bandwidth(final_coors, quantile=0.1)
# 	
# 	clust = cluster(final_coors, method="dbscan")
# 	clustModel = clust.clusterData()
# 	
# 	clusters=clust.getClusters()
# 	custSizes=clust.clustersSize()
# 	
# 	filtered_points=[]
# 	for k in custSizes:
# 		if k[1] > 20:
# 			my_members = clusters[:,3] == k[0]
# 			filtered_points.append(np.c_[final_coors[my_members],np.repeat([k[0]], len(final_coors[my_members]))])
# 	
# 	filtered_points=np.vstack(filtered_points)
# 	# plot the clusters
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(*filtered_points[:,0:3].T, s=2, c=filtered_points[:,3])

	
	return elecPointCloudsStruct, brainMask, voxsize

#%%

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture

class cluster():
	def __init__(self, pcd_list=None, method=None):
		self.pcd_list = pcd_list
		
		method_dic={
			"kmeans":{
				'method': KMeans,
				'options':{
					'init':'random',
					'n_clusters':15,
					'n_jobs':3,
					'n_init':10
					}
				},
			"affinity":{
				'method': AffinityPropagation,
				'options':{
					'damping':0.5,
					'max_iter':250,
					'affinity':'euclidean'
					}
				},
			"dbscan":{
				'method': DBSCAN,
				'options':{
					'eps':2.5,
					'min_samples':10
					}
				},
			"GaussianMixture":{
				'method': GaussianMixture,
				'options':{
					'n_components':7,
					'init_params':'kmeans'
					}
				},
			"MeanShift":{
				'method':MeanShift,
				'options':{
					'bandwidth': 30,
					'bin_seeding': True
					}
				}
			}
		
		self.method = method_dic[method]
		
	def clusterData(self):
		"""
		Take an input array of 3D points (x,y,z) and cluster based on the desired method.
		pcd_list is a list of points
		method_str is a string describing the method to use to cluster
		options is the options dictionary used by some scikit clustering
		functions
		"""
		
		# Keep two dimensions only to compute cluster (remove elevation)
		pcd_list = np.array(self.pcd_list)[:,:2]
	
		# Build the estimator with the given options
		self.estimator =self.method['method'](**self.method['options'])
	
		# Fit the estimator
		self.estimator.fit(self.pcd_list)
		
		return self.estimator
	
	def getClusters(self):
		# Get the labels and return the labeled points
		labels = self.estimator.labels_
		self.clusters = np.append(self.pcd_list, np.array([labels]).T, 1)
		
		return self.clusters
		
	def getCluster(self, i):
		"""
		Return the points belonging to a given cluster.
		clusters: list of labeled points where labels are the cluster ids
		i: id of the cluster to get
		label_field: field where the cluster id is stored in the list of labeled
		points
		"""
		return [c.tolist() for c in self.clusters if c[2] == i]

	def clustersSize(self):
		"""
		Return the size of the clusters given a list of labeled points.
		"""
		from collections import Counter
		
		self.getClusters()
		
		labels = self.clusters[:,2]
		counter = Counter(labels)
		
		return counter.most_common()

class IndexTracker:
	def __init__(self, ax, X,points=None, rotate=False):
		self.ax = ax
		self.points = points
		
		if rotate:
			self.X = np.fliplr(np.rot90(X,3))
			#if self.points is not None:
			#	self.points = np.rot90(points, k=1)
		else:
			self.X = X
		
		rows, cols, self.slices = X.shape
		self.ind = self.slices//2
		if self.points is not None:
			point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
			self.scatter,=ax.plot(point_plot[:,1],point_plot[:,0], marker="o", markersize=12, c = 'yellow', fillstyle='none', markeredgewidth=1, linestyle = 'None')
		self.im = ax.imshow(self.X[:, :, self.ind], origin='lower')
		self.update()

	def on_scroll(self, event):
		print("%s %s" % (event.button, event.step))
		if event.button == 'up':
			self.ind = (self.ind + 1) % self.slices
		else:
			self.ind = (self.ind - 1) % self.slices
		self.update()

	def update(self):
		
		if self.points is not None:
			#self.scatter.set_offsets(np.c_[self.points[self.points[:,2]==int(self.ind),1],self.points[self.points[:,2]==int(self.ind),0]])
			point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
			self.scatter.set_xdata(point_plot[:,1])
			self.scatter.set_ydata(point_plot[:,0])
		self.im.set_data(self.X[:, :, self.ind])
		self.ax.set_ylabel('slice %s' % self.ind)
		self.im.axes.figure.canvas.draw()


#%%


niiCTFile = r'/media/veracrypt6/projects/stealthMRI/imaging/clinical/deriv/dbsguide/sub-P223/source/sub-P223_ses-post_acq-Electrode_run-01_ct.nii.gz'
niiCTFile = r'/media/veracrypt6/projects/iEEG/imaging/clinical/bids/sub-P059/ses-post/ct/sub-P059_ses-post_acq-Electrode_run-01_ct.nii.gz'
niiCTFile = r'/home/greydon/Downloads/BRW_frame.nii.gz'

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", dest="finalDegree", default=3, required=False)
argParser.add_argument("-p", dest="displayProfiles", default=False, required=False)
argParser.add_argument("-dis", dest="displayMPR", default=False, required=False)
argParser.add_argument("-mask", dest="noMask", default=False, required=False)
argParser.add_argument("-bmask",dest='brainMask', default=False, required=False)
argParser.add_argument("-thres",dest='metalThreshold', default=800, required=False)
argParser.add_argument("-rev", dest="reverseDir", default=False, required=False)
argParser.add_argument("-det", dest="contactDetectionMethod", default='contactAreaCenter', required=False)
argParser.add_argument("-electype", dest='electrodeType', default=[], required=False)
argParser.add_argument("-medxml", dest="medtronicXMLPlan", default='', required=False)
argParser.add_argument("-egeom", dest="electrodeGeometries", default='/home/greydon/Documents/MATLAB/PaCER-master/res/electrodeGeometries.mat', required=False)

args = argParser.parse_args()


args.seegData = False
args.display = True

if not args.electrodeType:
	args.contactDetectionMethod = 'contactAreaCenter'


idx=list(elecPointCloudsStruct.keys())[0]
iElec=elecPointCloudsStruct[idx]



elecPointCloudsStruct, brainMask, voxsize = extractElectrodePointclouds(niiCTFile, args)
args.voxsize=voxsize

elecModels = {}
intensityProfiles = {}
skelSkelmms = {}
for idx, iElec in elecPointCloudsStruct.items():
	initialR3polynomial,_,_,_ = electrodePointCloudModelEstimate(iElec['pointCloudWorld'], args)
	elecModels[idx], intensityProfiles[idx], skelSkelmms[idx] = refitElec(initialR3polynomial, iElec['pointCloudWorld'], iElec['pixelValues'], args)
	


#%%


def getFloatCoordinate(point3dActive, strName):
	targetPoint3DElement = point3dActive.getElementsByTagName('point3dActive').item[0]
	coordElement = targetPoint3DElement.getElementsByTagName(strName).item[0]
	coordFloat = coordElement.getElementsByTagName('float').item[0]
	floatval = float(coordFloat.getFirstChild.getData)
	return floatval

def get3DFloatCoordinate(point3dActive):
	x = getFloatCoordinate(point3dActive, 'x')
	y = getFloatCoordinate(point3dActive, 'y')
	z = getFloatCoordinate(point3dActive, 'z')
	float3 = np.r_[x,y,z]
	return float3

def readMedtronicXMLTrajectory(filepath):
	xml = ET.parse(filepath).getroot()
	config = {}
	configVersion = xml.getElementsByTagName['dtd-version'].item[0].getFirstChild.getData
	
	if configVersion=='2.0':
		acPc = xml.getElementsByTagName('reformatSettings_v2').item[0]
		surgicalPlans = xml.getElementsByTagName('surgicalPlan_v2');
		
		if surgicalPlans.getLength() == 0:
			surgicalPlans = xml.getElementsByTagName('surgicalPlan')

		m1Element = acPc.getElementsByTagName('midline1').item[0]
		config['M1'] = get3DFloatCoordinate(m1Element);
		
		m2Element = acPc.getElementsByTagName('midline2').item[0]
		config['M2'] = get3DFloatCoordinate(m2Element);
		
		m3Element = acPc.getElementsByTagName('midline3').item[0]
		config['M3'] = get3DFloatCoordinate(m3Element);
		
	elif configVersion=='1.0':
		acPc = xml.getElementsByTagName('ACPC').item(0)
		surgicalPlans = xml.getElementsByTagName('surgicalPlan')
		
		m1Element = acPc.getElementsByTagName('midline').item[0]
		config['M1'] = get3DFloatCoordinate(m1Element)
		
		m2Element = acPc.getElementsByTagName('midline').item[1]
		config['M2'] = get3DFloatCoordinate(m2Element)
		
		m3Element = acPc.getElementsByTagName('midline').item[2]
		config['M3'] = get3DFloatCoordinate(m3Element)
		
		frameRods = xml.getElementsByTagName('frameRods');
		frameRod = frameRods.item[0]
		
		config['rods']={}
		for i in range(8):
			rod = frameRod.getElementsByTagName('rod').item[i]
			config['rods'][i+1]={}
			config['rods'][i+1]['coord'] = get3DFloatCoordinate(rod)
	
	acElement = acPc.getElementsByTagName('AC').item[0]
	config['AC'] = get3DFloatCoordinate(acElement)

	pcElement = acPc.getElementsByTagName('PC').item[0]
	config['PC'] = get3DFloatCoordinate(pcElement)
	
	noTraject = surgicalPlans.getLength()
	
	config['trajects']={}
	for i in range(noTraject-1):
		surgicalPlan = surgicalPlans.item[i]
		planNameElement = surgicalPlan.getElementsByTagName('name').item[0]
		config['trajects'][i+1].name = char(planNameElement.getFirstChild.getData)
		
		targetElement = surgicalPlan.getElementsByTagName('target').item[0]
		config['trajects'][i+1].target = get3DFloatCoordinate(targetElement)
		
		entryElement = surgicalPlan.getElementsByTagName('entry').item[0]
		config['trajects'][i+1].entry  = get3DFloatCoordinate(entryElement)
	
	return config


def getMostPropableAssociatedXmlDefinition(pointCloud):
	noSteps = 50
	dists = NaN(nReportedElecs,50)
	for j in range(nReportedElecs):
		entry = convertMedtronicCoordToLPI(reportedElecs.trajects(j).entry, niiCT).T
		target = convertMedtronicCoordToLPI(reportedElecs.trajects(j).target, niiCT).T
		direct = (target - entry)
		length = norm(target - entry)
		direct = direct / length
		
		steps = 0:noSteps
		stepLen = length / noSteps
		sampleOn = bsxfun(@plus,((stepLen .* steps).T * direct), entry)
		
		_, dist = dsearchn(pointCloud, sampleOn);
		
		dists[j,:] = dist
	
	_, idx = np.sort(np.mean(dists**2, 1))
	trajNo = idx[0]
	
	return trajNo



def getSliceCoords(point_data, targetZ):
	
	dist_vec=np.array([np.abs(x[2]-targetZ) for x in point_data])
	zSlice=[i for i,x in enumerate(dist_vec) if x <1]
	points = point_data[zSlice]
	
	return points[:,[0,1]]


class IndexTracker:
	def __init__(self, ax, X):
		self.ax = ax
		ax.set_title('use scroll wheel to navigate images')

		self.X = X
		rows, cols, self.slices = X.shape
		self.ind = self.slices//2

		self.im = ax.imshow(self.X[:, :, self.ind])
		self.update()

	def on_scroll(self, event):
		print("%s %s" % (event.button, event.step))
		if event.button == 'up':
			self.ind = (self.ind + 1) % self.slices
		else:
			self.ind = (self.ind - 1) % self.slices
		self.update()

	def update(self):
		self.im.set_data(self.X[:, :, self.ind])
		self.ax.set_ylabel('slice %s' % self.ind)
		self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax,threImg)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()

fig, ax = plt.subplots(1, 1)
tracker2 = IndexTracker(ax, brainMask)
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
plt.show()


values='pixelIdxs'
niiCT = nb.load(niiCTFile)
img_data=niiCT.get_fdata()

min_val = img_data.min()
max_val = img_data.max()
n_x, n_y, _ = img_data.shape
colormap = plt.cm.gray

z_slice=100

z_cut = img_data[:,:,z_slice]
X, Y = np.mgrid[0:n_x, 0:n_y]
Z = z_slice * np.ones((n_x, n_y))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=colormap((z_cut-min_val)/(max_val-min_val)))
for idx, item in elecPointCloudsStruct.items():
	print(np.unique(item['labels']))
	ax.scatter(item[values][:,0], item[values][:,1], item[values][:,2],s=2)

ax.set_xlim([0,img_data.shape[0]])
ax.set_ylim([0,img_data.shape[1]])
ax.set_zlim([0,img_data.shape[2]])
			

def sample_stack(img_data, elecPointCloudsStruct, rows=4, cols=4):
	if isinstance(img_data, nb.Nifti1Image):
		img_data = img_data.get_fdata()
	
	start=min(np.vstack((elecPointCloudsStruct[0][values][:,[2]], elecPointCloudsStruct[1][values][:,[2]])))
	end=max(np.vstack((elecPointCloudsStruct[0][values][:,[2]], elecPointCloudsStruct[1][values][:,[2]])))
	
	img_data_swap=np.fliplr(np.rot90(img_data, 3))
	z_index=np.linspace(start,end,rows*cols).astype(int)
	fig,ax = plt.subplots(rows,cols,figsize=[18,14])
	cnt = 0
	for i in range(rows*cols):
		if z_index[cnt] >= np.size(img_data_swap,2):
			fig.delaxes(ax[int(i/rows),int(i % rows)])
		else:
			ax[int(i/rows),int(i % rows)].set_title('Z axis %d' % int(z_index[cnt]))
			ax[int(i/rows),int(i % rows)].imshow(img_data_swap[:,:,int(z_index[cnt])],cmap='gray',interpolation=None, origin='lower')
			
			
			if 0 in list(elecPointCloudsStruct):
				point_data=getSliceCoords(elecPointCloudsStruct[0][values], int(z_index[cnt]))
				if point_data.size:
					point_data=point_data.mean(axis=0)
					ax[int(i/rows),int(i % rows)].plot(point_data[0], point_data[1],marker="o", markersize=10, c = 'orange', fillstyle='none', markeredgewidth=1)
			if 1 in list(elecPointCloudsStruct):
				point_data=getSliceCoords(elecPointCloudsStruct[1][values], int(z_index[cnt]))
				if point_data.size:
					point_data=point_data.mean(axis=0)
					ax[int(i/rows),int(i % rows)].plot(point_data[0], point_data[1],marker="o", markersize=10, c = 'yellow', fillstyle='none', markeredgewidth=1)
	
			#ax[int(i/rows),int(i % rows)].axis('off')
			cnt +=1
	
	plt.show()


sample_stack(convHullBrainMask)

niiCT = nb.load(niiCTFile)
img_data=niiCT.get_fdata()

min_val = img_data.min()
max_val = img_data.max()
n_x, n_y, _ = img_data.shape
colormap = plt.cm.gray

z_slice=100

z_cut = img_data[:,:,z_slice]
X, Y = np.mgrid[0:n_x, 0:n_y]
Z = z_slice * np.ones((n_x, n_y))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=colormap((z_cut-min_val)/(max_val-min_val)))

point_data=elecPointCloudsStruct[0][values]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if 0 in list(elecPointCloudsStruct):
	point_data=elecPointCloudsStruct[0][values]
	ax.scatter(point_data[:,0], point_data[:,1],point_data[:,2], c = 'red')
if 1 in list(elecPointCloudsStruct):
	point_data=elecPointCloudsStruct[1][values]
	ax.scatter(point_data[:,0], point_data[:,1],point_data[:,2], c = 'blue')
ax.imshow(img_data[:,:,60],cmap='gray')




min_val = img_data.min()
max_val = img_data.max()
n_x, n_y, _ = img_data.shape
colormap = plt.cm.gray

z_slice=100

z_cut = img_data[:,:,z_slice]
X, Y = np.mgrid[0:n_x, 0:n_y]
Z = z_slice * np.ones((n_x, n_y))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=colormap((z_cut-min_val)/(max_val-min_val)))
if 0 in list(elecPointCloudsStruct):
	point_data=elecPointCloudsStruct[0][values]
	ax.scatter(point_data[:,0], point_data[:,1],point_data[:,2], c = 'red')
if 1 in list(elecPointCloudsStruct):
	point_data=elecPointCloudsStruct[1][values]
	ax.scatter(point_data[:,0], point_data[:,1],point_data[:,2], c = 'blue')
ax.invert_zaxis()

ax.set_title("z slice")
ax.view_init(elev=50, azim=44)


import ants
from nilearn import plotting,image

from nilearn.datasets import load_mni152_template

template = ants.image_read(ants.get_ants_data('mni'))

ct_ants = ants.image_read(niiCTFile)
ct_ants_reg = ants.registration(template, ct_ants, type_of_transform='QuickRigid')
ct_ants_reg_applied=ants.apply_transforms(template, ct_ants, transformlist=ct_ants_reg['fwdtransforms'])
ct_resample = ants.to_nibabel(ct_ants_reg_applied)

img = nb.load(niiCTFile)
mask_data=elecsPointcloudStruct[0]['binaryMaskImage']
nii_mask = nb.Nifti1Image(mask_data, ct_resample.affine, ct_resample.header)
mask_ants=ants.from_nibabel(nii_mask)

mask_ants_reg_applied = ants.apply_transforms(ct_ants_reg_applied, mask_ants, transformlist=ct_ants_reg['fwdtransforms'])
mask_resample = ants.to_nibabel(mask_ants_reg_applied)

mask_params = {
				'symmetric_cmap': True,
				'cut_coords':[0,0,0],
				'dim': 1,
				'cmap':'viridis',
				'opacity':0.7
				}

html_view = plotting.view_img(stat_map_img=mask_resample,bg_img=ct_resample, **mask_params)

html_view.save_as_html(snakemake.output.html)

d = {'x': point_data[:,0], 'y': point_data[:,1], 'z': point_data[:,2]}
pts=pd.DataFrame(data=d)
points_ants_reg_applied=ants.apply_transforms_to_points(3, pts, ct_ants_reg['fwdtransforms'])


html_view = plotting.view_markers(points_ants_reg_applied.values)
html_view.open_in_browser()

img = nb.load(niiCTFile)

binMask=largestConnCompSliceWise(img.get_fdata())

mask_data=maskedImg.copy()
mask_data[brainMask] = 1
mask_data[~brainMask] = 0
nii_mask = nb.Nifti1Image(maskedImg, niiCT.affine, niiCT.header)

template = load_mni152_template()
ref_resamp = image.resample_img(niiCT, target_affine=template.affine, interpolation='nearest')
flo_resamp = image.resample_to_img(nii_mask, ref_resamp)

mask_params = {
				'symmetric_cmap': True,
				'cut_coords':[0,0,0],
				'dim': 1,
				'cmap':'viridis',
				'opacity':0.7
				}

html_view = plotting.view_img(stat_map_img=nii_mask,bg_img=niiCT,**mask_params)
html_view.open_in_browser()































	
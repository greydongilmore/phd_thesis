#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:52:44 2021

@author: greydon
"""
import vtk, numpy as np, pandas as pd,os, shutil, csv, json, sys, subprocess, platform,math
import nibabel as nb
from skimage import morphology,measure
from skimage.filters import threshold_otsu
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math
import seaborn as sns

class IndexTracker:
	def __init__(self, ax, img_data,points=None, rotate_img=False, rotate_points=False, title=None):
		self.ax = ax
		self.scatter=None
		self.points=None
		self.title=title
		no_index= True
		if rotate_img:
			self.img_data = np.fliplr(np.rot90(img_data.copy(),3))
			#if self.points is not None:
			#	self.points = np.rot90(points, k=1)
		else:
			self.img_data = img_data.copy()
		
		rows, cols, self.slices = self.img_data.shape
		self.ind = self.slices//2
		if points is not None:
			self.points=points.copy()
			if rotate_points:
				self.points[:,[0,1]]=self.points[:,[1,0]]
			while no_index==True:
				if any(self.points[:,2]==self.ind):
					no_index=False
					point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
					self.scatter,=ax.plot(point_plot[:,1],point_plot[:,0], marker="o", markersize=12, c = 'yellow', fillstyle='none', markeredgewidth=1, linestyle = 'None')
				else:
					self.ind+=1
				
		self.im = ax.imshow(self.img_data[:, :, self.ind], origin='lower')
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
			if any(self.points[:,2]==self.ind):
				point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
				self.scatter.set_xdata(point_plot[:,1])
				self.scatter.set_ydata(point_plot[:,0])
		self.im.set_data(self.img_data[:, :, self.ind])
		
		if self.title is not None:
			plot_title=self.title
		else:
			plot_title='slice %s' % self.ind
		
		self.ax.set_title(plot_title, fontdict={'fontsize': 18, 'fontweight': 'bold'})
		self.ax.tick_params(axis='x', labelsize=14)
		self.ax.tick_params(axis='y', labelsize=14)
		self.im.axes.figure.canvas.draw()

#%%

def determineImageThreshold( img_data):
	
	hist_y, hist_x = np.histogram(img_data.flatten(), bins=256)
	hist_x = hist_x[0:-1]
	
	cumHist_y = np.cumsum(hist_y.astype(float))/np.prod(np.array(img_data.shape))
	# The background should contain half of the voxels
	minThreshold_byCount = hist_x[np.where(cumHist_y > 0.90)[0][0]]
	
	hist_diff = np.diff(hist_y)
	hist_diff_zc = np.where(np.diff(np.sign(hist_diff)) == 2)[0].flatten()
	minThreshold = hist_x[hist_diff_zc[hist_x[hist_diff_zc] > (minThreshold_byCount)][0]]
	print(f"first maxima after soft tissue found: {minThreshold}")
	
	
	return minThreshold

niiFrameFile = r'/home/greydon/Documents/GitHub/trajectoryGuide/resources/frames/frame-leksellg_acq-clinical_run-01.nii.gz'
niiFrameFile = r'/home/greydon/Documents/GitHub/trajectoryGuide/resources/frames/frame-brw_acq-clinical_run-03.nii.gz'


img_obj = sitk.ReadImage(niiFrameFile)
img_obj = sitk.PermuteAxes(img_obj, [2,1,0])
img_data = sitk.GetArrayFromImage(img_obj)
voxsize = img_obj.GetSpacing()

minThreshold=determineImageThreshold(img_data)
thresh_img = img_obj > minThreshold

stats = sitk.LabelShapeStatisticsImageFilter()
stats.SetComputeOrientedBoundingBox(True)
stats.Execute(thresh_img)

connectedComponentImage = sitk.ConnectedComponent(thresh_img, False)
stats.Execute(connectedComponentImage)

labelBBox_size_mm = np.array([stats.GetOrientedBoundingBoxSize(l) for l in stats.GetLabels()])*voxsize
labelBBox_center_mm = np.array([np.array(stats.GetOrientedBoundingBoxOrigin(l)) + np.dot(np.reshape(stats.GetOrientedBoundingBoxDirection(l), [3,3]),
								(np.array(stats.GetOrientedBoundingBoxSize(l))*voxsize)/2  ) for l in stats.GetLabels()])

allObjects_bboxCenter = np.array(stats.GetCentroid(1))
allObjects_bboxSize = np.array(stats.GetOrientedBoundingBoxSize(1))*voxsize

labelBBoxCentroidDistFromCenter = np.linalg.norm(labelBBox_center_mm - np.tile(allObjects_bboxCenter, [labelBBox_center_mm.shape[0], 1]), axis=1)
objectMajorAxisSize_mask = np.sum(labelBBox_size_mm > allObjects_bboxSize*0.5, axis=1 )>0

compacityRatio_mask = np.sum(np.stack([labelBBox_size_mm[:,0]/labelBBox_size_mm[:,1],
									   labelBBox_size_mm[:,1]/labelBBox_size_mm[:,2],
									   labelBBox_size_mm[:,2]/labelBBox_size_mm[:,0]], axis=1)>4, axis=1 )>0

centroidDist_mask =  labelBBoxCentroidDistFromCenter > np.min(allObjects_bboxSize)*0.3

print("%d objects left after objectMajorAxisSize_mask"%(np.sum(objectMajorAxisSize_mask==1)))
print("%d objects left after compacityRatio_mask"%(np.sum(compacityRatio_mask*objectMajorAxisSize_mask==1)))
print("%d objects left after centroidDist_mask"%(np.sum(centroidDist_mask*compacityRatio_mask*objectMajorAxisSize_mask==1)))

labelToKeep_mask = objectMajorAxisSize_mask * compacityRatio_mask * centroidDist_mask
connected_labelMap = sitk.LabelImageToLabelMap(connectedComponentImage)

label_renameMap = sitk.DoubleDoubleMap()
for i, toKeep in enumerate(labelToKeep_mask):
	if not toKeep:
		label_renameMap[i+1]=0

newLabelMap = sitk.ChangeLabelLabelMap(connected_labelMap, label_renameMap)
stats.Execute(sitk.LabelMapToLabel(newLabelMap))


coords = {label:np.where(sitk.GetArrayFromImage(connectedComponentImage) == label) for label in stats.GetLabels()}
component=[]
for icoord in coords:
	physical_points = np.stack([[int(x), int(y), int(z)] for x,y,z in zip(coords[icoord][0], coords[icoord][1], coords[icoord][2])])
	physical_intensity = np.stack([img_data[int(x), int(y), int(z)] for x,y,z in zip(coords[icoord][0], coords[icoord][1], coords[icoord][2])])
	component.append(np.c_[physical_points, np.repeat(icoord,len(physical_points)), physical_intensity])

component=np.vstack(component)


points = np.stack(sorted(component, key=(lambda k: k[2])))

AP_index=1
ML_index=0

# find distance from max in y-axis
distance_AP_max = abs(points[:, AP_index] - points[:,AP_index].max())
# identiy the gaps in distances greater than 9 voxels (these are seperate bars)
gaps = [[s, e] for s, e in zip(sorted(distance_AP_max), sorted(distance_AP_max)[1:]) if s+15 < e]
# pick the first gap for the anterior N-localizer
min_threshold=gaps[0][0]
AP_points=points[distance_AP_max <=min_threshold, :]
points=np.delete(points,np.where(distance_AP_max <= min_threshold)[0],0)
AP_points=np.c_[AP_points, [2]*len(AP_points)]

distance_ML_min = abs(points[:,ML_index] - points[:,ML_index].min())
gaps = [[s, e] for s, e in zip(sorted(distance_ML_min), sorted(distance_ML_min)[1:]) if s+3 < e]
# pick the first gap for the anterior N-localizer
min_threshold=gaps[0][0]
ML1_points=points[distance_ML_min <= min_threshold, :]
if len(ML1_points) <1000 and len(gaps) > 1:
	min_threshold=gaps[0][0]
	ML1_points=points[(distance_ML_min > gaps[0][0]) & (distance_ML_min <= gaps[1][0]), :]
	points=np.delete(points,np.where((distance_ML_min > gaps[0][0]) & (distance_ML_min <= gaps[1][0]))[0],0)
else:
	points=np.delete(points,np.where(distance_ML_min <= min_threshold)[0],0)

ML1_points=np.c_[ML1_points, [1]*len(ML1_points)]

distance_ML_max = abs(points[:,ML_index] - np.median(points[:,ML_index]))
gaps = [[s, e] for s, e in zip(sorted(distance_ML_max), sorted(distance_ML_max)[1:]) if s+10 < e]
# since only points associated with the last localizer should be present the gaps list should be empty
# if it's not empty then pick only the points meeting the threshold, the remainder are noise
if gaps:
	# pick the first gap for the anterior N-localizer
	min_threshold=gaps[0][0]
	ML2_points=points[distance_ML_max <= min_threshold, :]
else:
	ML2_points=points[:,:]

ML2_points=np.c_[ML2_points, [3]*len(ML2_points)]

component=np.r_[AP_points, ML1_points,ML2_points]


combined=[]
for label in np.unique(component[:,-1]):
	sort_idx=0
	if component[component[:,-1]==label,0].std() < component[component[:,-1]==label,1].std():
		sort_idx=1
	clust_tmp=[]
	for ind in np.unique(component[(component[:,-1]==label),2]):
		points=component[(component[:,-1]==label)&(component[:,2]==ind),:]
		points = np.stack(sorted(points, key=(lambda k: k[sort_idx])))
		gaps = [[s, e] for s, e in zip(points[:,sort_idx], points[:,sort_idx][1:]) if s+3 < e]
		if len(gaps)==2:
			bar_1=points[points[:,sort_idx]<=gaps[0][0],:]
			bar_2=points[(points[:,sort_idx]>=gaps[0][1])&(points[:,sort_idx]<=gaps[1][0]),:]
			bar_3=points[points[:,sort_idx]>=gaps[1][1],:]
			
			if label==np.unique(component[:,-1]).min():
				label_start=1
			elif label not in (np.unique(component[:,-1]).min(), np.unique(component[:,-1]).max()):
				label_start=4
			elif label==np.unique(component[:,-1]).max():
				label_start=7
			
			clust_tmp.append(np.vstack((np.c_[bar_1,np.repeat(label_start,len(bar_1))],
						   np.c_[bar_2,np.repeat(label_start+1,len(bar_2))],
						   np.c_[bar_3,np.repeat(label_start+2,len(bar_3))])))
	if clust_tmp:
		mask=np.vstack(clust_tmp)
		combined.append(mask)
	
	#frame_bot=np.array([np.mean(mask[:,2])-61,mask[:,2].min()]).max()
	#frame_top=np.array([np.mean(mask[:,2])+61,mask[:,2].max()]).min()
	#combined.append(mask[(mask[:,2]>=frame_bot) & (mask[:,2]<=frame_top),:])

combined=np.vstack(combined)
combined[:,3]=combined[:,-1]
combined = combined[:,list(range(combined.shape[1]-1))]

points = np.c_[np.stack([img_obj.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) for z,y,x in zip(combined[:,0], combined[:,1], combined[:,2])]),combined[:,3:]]
points = np.stack(sorted(points, key=(lambda k: k[2])))

point_Ras=[]
point_Ras_int=[]
intensity_mean=False
for islice in np.unique(points[:, 2]):
	for ilabel in np.unique(points[(points[:,2]==islice), -1]):
		npoints=len(points[(points[:,2]==islice)&(points[:,-1]==ilabel), :2])
		
		cluster_data=points[(points[:,2]==islice)&(points[:,-1]==ilabel), :]
		sum_pixel_values = sum(cluster_data[:,4])
		x_iw = sum(np.array(cluster_data[:,0]) * np.array(cluster_data[:,4])) / sum_pixel_values
		y_iw = sum(np.array(cluster_data[:,1]) * np.array(cluster_data[:,4])) / sum_pixel_values
		z_iw = sum(np.array(cluster_data[:,2]) * np.array(cluster_data[:,4])) / sum_pixel_values
		position_ijk=np.array([x_iw, y_iw, z_iw])
		
		point_Ras_int.append(np.hstack((position_ijk[:3], ilabel, np.mean(points[(points[:,2]==islice)&(points[:,-1]==ilabel), 4]),npoints)))


combined_final=np.vstack(point_Ras_int)


plt_img=sitk.GetArrayFromImage(thresh_img)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, plt_img, points=None,rotate_img=True, rotate_points=True)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()

plt_img=sitk.GetArrayFromImage(out_img)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, plt_img, points=None,rotate_img=True, rotate_points=True)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()


fig = plt.figure(figsize=(16,14))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(combined[:,0], combined[:,1],combined[:,2], c = combined[:,3],s=6)
ax.tick_params(axis='x', labelrotation=45)
ax.tick_params(axis='y', labelrotation=45)
ax.set_xlabel('X axis: M-L (mm)', fontsize=14, fontweight='bold', labelpad=14)
ax.set_ylabel('Y axis: A-P (mm)', fontsize=14, fontweight='bold', labelpad=14)
ax.set_zlabel('Z axis: I-S (mm)', fontsize=14, fontweight='bold', labelpad=14)
ax.xaxis._axinfo['label']['space_factor'] = 3.8
ax.view_init(elev=20, azim=50)
ax.figure.canvas.draw()
fig.tight_layout()




cc_img = sitk.GetArrayFromImage(out_img)

fig, ax = plt.subplots(1, 1)
tracker2 = IndexTracker(ax, cc_img, points=None,rotate_img=True)
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
plt.show()



stats.GetLabels()




fig, ax=plt.subplots(2,1, sharex=True)
ax[0].plot(hist_x[0:-1], hist_y, label='intensity histogram')
ax[0].plot(hist_x, hist_y, label='intensity histogram')
ax[1].plot(hist_x, cumHist_y, label='cumulative histogram')
ax[1].axvline(x=minThreshold_byCount, ymin=0.0, ymax=1.0, color='r')

ax[0].plot(hist_x[1:], hist_diff, label='derivative of intensity histogram')
ax[0].plot(hist_x[hist_diff_zc], hist_diff[hist_diff_zc], linestyle='', marker='o', label="zero crossings of derivative of intensity histogram")

ax[0].axvline(x=minThreshold_byCount, ymin=0.0, ymax=1.0, color='r')
ax[0].axvline(x=minThreshold_byVariation, ymin=0.0, ymax=1.0, color='g')
fig.legend()
ax[0].grid()

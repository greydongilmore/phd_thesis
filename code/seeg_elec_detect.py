#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:01:39 2021

@author: greydon
"""

import nibabel as nib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from skimage import color
import torch
import numpy as np
from torchvision import utils
from nilearn import plotting,image
from sklearn.decomposition import PCA
from nilearn.datasets import load_mni152_template
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import median_filter
from skimage import morphology
from skimage import measure,segmentation
from skimage import feature
from scipy import ndimage


anat_nii=r'/home/greydon/Documents/data/DBS/derivatives/fastsurfer/sub-P237/fastsurfer/mri/T1.mgz'
img=r'/home/greydon/Documents/data/DBS/bids/sub-P237/ses-post/ct/sub-P237_ses-post_acq-Electrode_run-01_ct.nii.gz'
pred=r'/home/greydon/Documents/data/DBS/derivatives/fastsurfer/sub-P237/fastsurfer/mri/aparc.DKTatlas+aseg.deep.mgz'


img_obj = nib.load(img)
pred_data = nib.load(pred).get_fdata()


Img = img_obj.get_fdata().copy() / 255
MaxElecs =4

ElecRad = 1.3
ProjSurfRaw = app.ProjSurfRaw;

TMax = (app.CTRng(4)-app.CTRng(1))/(app.CTRng(2)-app.CTRng(1)); %max (CTRng(4)) normalized with respect to 1st (CTRng(1)) and 99th (CTRng(2)) percentiles
TMin = 1; %99th percentile
Thresh = linspace(TMax,TMin,21); Thresh(1) = []; %start with 1 step below max and progressively lower threshold to search for electrodes
ThreshHU = Thresh*(app.CTRng(2)-app.CTRng(1))+app.CTRng(1); %thresholds in raw units
ThreshHU = (ThreshHU-app.CTInfo.pinfo(2))./app.CTInfo.pinfo(1); %convert to hounsfield

ElecVolVox = ceil(4/3*pi*mean(ElecRad./(img_obj.header.get_zooms())).^3); %approximate volume of an electrode (standard ecog - typically largest intracranial electrode) in number of voxels
ElecVolRng = [6,ElecVolVox]; %6 voxels as minimum seems to work well for a wide range of electrode types

StartTime = tic;
CCList = cell(length(Thresh),1);
WCList = cell(length(Thresh),1);
for k=1:length(Thresh)
	app.WaitH.Value = k/length(Thresh);
	
	ImgBin = Img>Thresh(k);
	
	CC = bwconncomp(ImgBin,6);
	CCSize = cellfun(@length,CC.PixelIdxList);
	
	idx = CCSize<ElecVolRng(1)|CCSize>ElecVolRng(2);
	CC.PixelIdxList(idx) = [];
	CC.NumObjects = length(CC.PixelIdxList);
	
	if CC.NumObjects>0
		s = regionprops3(CC,Img.*ImgBin,'weightedcentroid','meanintensity');
				
		WC = s.WeightedCentroid;
		WC(:,[1,2]) = WC(:,[2,1]);
		
		WCmm = [WC,ones(size(WC,1),1)]*img_obj.affine'; WCmm(:,4) = [];
		
		idx = LeG_intriangulation(ProjSurfRaw.vertices,ProjSurfRaw.faces,WCmm);
		
		s(~idx,:) = [];
		WC(~idx,:) = [];
		CC.PixelIdxList(~idx) = [];
		CC.NumObjects = length(CC.PixelIdxList);
		
		[~,sidx] = sort([s.MeanIntensity],'descend');
		
		WC = round(WC(sidx,:));
		WCList(k) = {WC};
		CCList(k) = {CC};
	end
end
NumObj = cellfun(@(x)size(x,1),WCList);

cc = bwconncomp(abs(diff(NumObj))<=5 & NumObj(1:end-1)>10);%change in number of detected electrodes (as threshold decreases) should be less than 5 and total number greater than 10
skipflag = true;
if cc.NumObjects>0
	ccsize = cellfun(@length,cc.PixelIdxList);
	cc.PixelIdxList(ccsize<2) = []; %need at least 3 (2 diffs) stable thresholds where number of detected electrodes does not change by more than 5
	cc.NumObjects = length(cc.PixelIdxList);
	if cc.NumObjects>0
		ccval = cellfun(@(x)mean(NumObj(x)),cc.PixelIdxList);
		[~,midx] = max(ccval); %find the threshold segment with the largest number of electrodes
		idx = cc.PixelIdxList{midx};
%         [~,midx] = max(NumObj(idx)); %find the theshold with the largest number of electrodes within the chosen segment
%         idx = idx(midx);
		idx = idx(round(end/2)); %choose the middle index of cluster
		skipflag = false;
	end
end
if skipflag %if no stable clusters are found, do this
	idx = find(NumObj>10 & NumObj<MaxElecs);
	if ~isempty(idx)
		idx = idx(round(end/2));
	else
		idx = round(length(Thresh)/2);
	end
end

WC = WCList{idx};
THU = ThreshHU(idx);
T = Thresh(idx);

%outlier removal
pd = pdist2(WC,WC,'euclidean','smallest',2); %find closest electrode to each detected electrode
WC(pd(end,:)*mean(voxsize)<1,:) = []; %remove detections that are closer than 1mm
WC(MaxElecs+1:end,:) = []; %remove if more than 250 detections

%plotting results
fH = figure('position',[50,50,400,400],'name',app.PatientIDStr,'visible','off'); 
aH = axes('parent',fH);
plot(aH,ThreshHU,NumObj); 
hold(aH,'on');
plot(aH,ThreshHU(idx),NumObj(idx),'or');
EndTime = toc(StartTime);
xlabel('Thresh(HU)')
ylabel('#Elecs')
TMaxHU = (app.CTRng(4)-app.CTInfo.pinfo(2))./app.CTInfo.pinfo(1);
TMinHU = (app.CTRng(3)-app.CTInfo.pinfo(2))./app.CTInfo.pinfo(1);
title(aH,sprintf('%0.1fs, %0.0fhu (%0.0f,%0.0f)',EndTime,THU,TMinHU,TMaxHU)) %time, threshold(HU), min(HU), max(HU)
print(fH,fullfile(app.SaveDir,'AutoElec.png'),'-dpng','-r300')
close(fH);



fig, ax = plt.subplots(1, 1)
tracker1 = IndexTracker(ax, mask, points=None,rotate_img=False, rotate_points=False)
fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
plt.show()


def largestConnCompSliceWise(img3d):
	binaryMask = np.empty(img3d.shape, dtype=bool)
	for slice_idx in range(img3d.shape[2]):
		labels = measure.label(img3d[:,:,slice_idx])
		ccProps = measure.regionprops(labels)
		ccProps.sort(key=lambda x: x.area, reverse=True)
		areas=np.array([prop.area for prop in ccProps])
		
		binarySlice = np.empty(np.shape(img3d[:,:,slice_idx]), dtype=bool)
		if not areas.size==0:
			brainMaskIdxs = ccProps[0].coords
			binarySlice[tuple([brainMaskIdxs[:,0],brainMaskIdxs[:,1]])] = True
		binaryMask[:,:,slice_idx] = binarySlice
	return binaryMask

def extractBrainConvHull(img_obj):
	
	brain_ct_min = 20
	brain_ct_max = 60
	
	ctIso = img_obj.get_fdata()
	
	ctIso[ctIso < -1024] = -1024
	ctIso[ctIso > 3071] = 3071
	
	print('Applying median filter to raw data...')
	ctIso = median_filter(ctIso, size=(3,3,3))
	
	threshold_image = (ctIso > brain_ct_min)&(ctIso < brain_ct_max)
	
	structEle=morphology.ball(np.ceil(2 / max((img_obj.header.get_zooms()))))
	morphImg = morphology.binary_opening(threshold_image, structEle)
	
	labels, n_labels = measure.label(morphImg, background=0, return_num=True)
	label_count = np.bincount(labels.ravel().astype(np.int))
	label_count[0] = 0
	
	ccProperties = measure.regionprops(labels)
	ccProperties.sort(key=lambda x: x.area, reverse=True)
	areas=np.array([prop.area for prop in ccProperties])
	
	#mask = labels == label_count.argmax()
	
	mask = largestConnCompSliceWise(morphImg)
	
	masked_image = ctIso
	masked_image[~mask] = np.nan
	mask = (masked_image > brain_ct_min)&(masked_image < brain_ct_max)
	
	print('Extracting convex hull brain mask...')
	convHullBrainMask=flood_fill_hull(mask)
	
	print('Running binary erosion on brain convex hull...')
	convHullBrainMask = morphology.binary_erosion(convHullBrainMask, structEle)
	convHullBrainMask = morphology.binary_erosion(convHullBrainMask, structEle)
	
	return convHullBrainMask



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


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


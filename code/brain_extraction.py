#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 01:31:35 2021

@author: greydon
"""

from skimage import exposure
from skfuzzy import cmeans
import numpy as np
from skimage import filters
from skimage import morphology
from scipy import ndimage
import nibabel as nb
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_hist(ax, data, title=None):
	ax.hist(data.ravel(), bins=256)
	ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

	if title:
		ax.set_title(title)

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()

def __get_brain_coarse_mask(raw_slices):
	coarse_masks = []
	# Thresholding with Otsu's method
	threshold = filters.threshold_otsu(raw_slices)
	# Apply threshold to obtain binary slice
	coarse_masks = raw_slices > threshold

	coarse_masks = morphology.closing(
		np.array(coarse_masks), morphology.ball(2))

	coarse_masks = ndimage.binary_fill_holes(np.array(coarse_masks))
	for index in enumerate(coarse_masks):
		coarse_masks[index[0], :, :] = ndimage.binary_fill_holes(
			coarse_masks[index[0], :, :])

	return np.array(coarse_masks)





niiCTFile=r'/home/greydon/Documents/GitHub/blog/docs/image_processing/static/sub-P222_ses-pre_run-01_T1w.nii.gz'

img_obj = nb.load(niiCTFile)
voxel_dims = (img_obj.header["dim"])[1:4]
voxsize = (img_obj.header["pixdim"])[1:4]

img_data = img_obj.get_fdata().copy()
image_type=None

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(np.rot90(img_data[:, :, img_data.shape[-1]//2]), cmap='bone')

# compute and display the histogram.
pixelCount, grayLevels =exposure.histogram(img_data)

fig, a = plt.subplots(nrows=1, ncols=1)
plot_hist(a, img_data, title="Histogram of original image")

show_histogram(img_data)

# Threshold to create a binary image
mask_data = img_data > img_data.mean()

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(np.rot90(mask_data[:, :, mask_data.shape[-1]//2]), cmap='bone')

labels, n_labels = measure.label(mask_data, background=0, return_num=True)
label_count = np.bincount(labels.ravel().astype(np.int))
label_count[0] = 0

mask_data = labels == label_count.argmax()

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(np.rot90(mask_data[:, :, mask_data.shape[-1]//2]), cmap='bone')


n_clust=3

[t1_cntr, t1_mem, _, _, _, _, _] = cmeans(img_data[mask_data].reshape(-1, len(mask_data[mask_data])),n_clust, 2, 0.005, 100)
t1_mem_list = [t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])]  # CSF/GM/WM

mask = np.zeros(img_data.shape + (n_clust,))
for i in range(n_clust):
	mask[..., i][mask_data] = t1_mem_list[i]


fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(np.rot90(mask[:, :, mask.shape[2]//2,1]), cmap='bone')


skullImage=mask[:,:,:,1]>0.9
brainImage=mask[:,:,:,0]>0.9
csfImage=mask[:,:,:,2]>0.9

skullImage=skullImage+csfImage
# Get rid of small specks of noise
skullImage = morphology.remove_small_objects(skullImage, 10)
brainImage = morphology.remove_small_objects(brainImage, 20)

fig, (a,b) = plt.subplots(1,2,figsize=(15,5))
a.matshow(np.rot90(skullImage[:, :, skullImage.shape[-1]//2]), cmap='bone')
b.matshow(np.rot90(brainImage[:, :, brainImage.shape[-1]//2]), cmap='bone')



skullImage = ndimage.binary_fill_holes(skullImage)
brainImage = ndimage.binary_fill_holes(brainImage)

fig, (a,b) = plt.subplots(1,2,figsize=(15,5))
a.matshow(np.rot90(skullImage[:, :, skullImage.shape[-1]//2]), cmap='bone')
b.matshow(np.rot90(brainImage[:, :, brainImage.shape[-1]//2]), cmap='bone')




brainMorph = morphology.erosion(brainImage, np.ones((2,2,2)))
skullMorph = morphology.dilation(skullImage, np.ones((3,3,3)))

fig, (a,b) = plt.subplots(1,2,figsize=(15,5))
a.matshow(np.rot90(skullMorph[:, :, skullMorph.shape[-1]//2]), cmap='bone')
b.matshow(np.rot90(brainMorph[:, :, brainMorph.shape[-1]//2]), cmap='bone')


finalImage = img_data.copy()
finalImage[~brainMorph] = 0

finalImage2 = img_data.copy()
finalImage2[skullMorph] = 0

fig, (a,b,c) = plt.subplots(1,3,figsize=(15,5))
a.matshow(np.rot90(img_data[:, :, img_data.shape[-1]//2]), cmap='bone')
b.matshow(np.rot90(finalImage2[:, :, finalImage2.shape[-1]//2]), cmap='bone')
c.matshow(np.rot90(finalImage[:, :, finalImage.shape[-1]//2]), cmap='bone')



coarse_masks = __get_brain_coarse_mask(img_data)


fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(np.rot90(coarse_masks[:, :, coarse_masks.shape[2]//2]), cmap='bone')






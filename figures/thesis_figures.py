#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 21:52:23 2021

@author: greydon
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import colors
from skimage import morphology


class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)

def Rx(phi):
	return np.array([[1, 0, 0],
					 [0, np.cos(phi), -np.sin(phi)],
					 [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
	return np.array([[np.cos(theta), 0, np.sin(theta)],
					 [0, 1, 0],
					 [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
	return np.array([[np.cos(psi), -np.sin(psi), 0],
					 [np.sin(psi), np.cos(psi), 0],
					 [0, 0, 1]])


fontTitle = {
	'family': 'DejaVu Sans',
	'weight': 'bold',
	'size': 22,
}


fontAxis = {
	'family': 'DejaVu Sans',
	'color':  'black',
	'weight': 'bold',
	'size': 16
}


out_path='/media/greydon/KINGSTON34/phdCandidacy/thesis/imgs'

arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
text_options = {'horizontalalignment': 'center',
				'verticalalignment': 'center',
				'fontsize': 15,
				'fontweight': 'bold'}
max_val=1

#%% Figure 2.2: Coordinate systems

# reference https://stackoverflow.com/a/29188796

# define origin
o = np.array([0,0,0])

# define ox0y0z0 axes
x0 = np.array([1,0,0])
y0 = np.array([0,1,0])
z0 = np.array([0,0,1])

# define ox1y1z1 axes
psi = 45 * np.pi / 180
x1 = Rz(psi).dot(x0)
y1 = Rz(psi).dot(y0)
z1 = Rz(psi).dot(z0)


fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(121, projection='3d')

a=Arrow3D([o[0], x0[0]], [o[1], x0[1]], [o[2], x0[2]], **arrow_prop_dict)
ax.add_artist(a)
a=Arrow3D([o[0], y0[0]], [o[1], y0[1]], [o[2], y0[2]], **arrow_prop_dict)
ax.add_artist(a)
a=Arrow3D([o[0], z0[0]], [o[1], z0[1]], [o[2], z0[2]], **arrow_prop_dict)
ax.add_artist(a)
ax.set_xlim([o[0], max_val])
ax.set_ylim([o[0], max_val])
ax.set_zlim([o[0], max_val])

# plot ox1y1z1 axes
ax.plot(x1[0], y1[1], z1[2], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)
ax.plot([x1[0], x1[0]], [y1[1], y1[1]], [o[0], z1[2]], 'k--', linewidth=1)
ax.plot([x1[0], x1[0]], [o[0], y1[1]], [o[0], o[0]], 'k--', linewidth=1)
ax.plot([o[0], x1[0]], [y1[1], y1[1]], [o[0], o[0]], 'k--', linewidth=1)

ax.text(1.2*x0[0],1.2*x0[1],1.2*x0[2],r'$x$', **text_options)
ax.text(1.1*y0[0],1.1*y0[1],1.1*y0[2],r'$y$', **text_options)
ax.text(1.1*z0[0],1.1*z0[1],1.1*z0[2],r'$z$', **text_options)
ax.text(0.0,0.0,-0.05,r'$o$', **text_options)
ax.text(x1[0]+.12,y1[1]/2, 0, r'$y$', **text_options)
ax.text(x1[0]/2,y1[1]+0.07, 0, r'$x$', **text_options)
ax.text(x1[0]+.05, y1[1]+.05, z0[2]/1.9, r'$z$', **text_options)
ax.text(x0[0]-.03, y0[1]-.03, z0[2]+.05, r'$P(x,y,z)$', **text_options)

ax.set_title('Cartesian coordinate system', fontdict={'fontsize': 18, 'fontweight': 'bold'})

ax.view_init(elev=10, azim=22)
ax.dist = 10
ax.set_axis_off()
ax.grid(True)


# define ox2y2z2 axes
theta = 45 * np.pi / 180
x2 = Rz(psi).dot(Ry(theta)).dot(x0)
y2 = Rz(psi).dot(Ry(theta)).dot(y0)
z2 = Rz(psi).dot(Ry(theta)).dot(z0)

phi = 45 * np.pi / 180
x3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(x0)
y3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(y0)
z3 = Rz(psi).dot(Ry(theta)).dot(Rx(phi)).dot(z0)


ax = fig.add_subplot(122, projection='3d')

a=Arrow3D([o[0], x0[0]], [o[1], x0[1]], [o[2], x0[2]], **arrow_prop_dict)
ax.add_artist(a)
a=Arrow3D([o[0], y0[0]], [o[1], y0[1]], [o[2], y0[2]], **arrow_prop_dict)
ax.add_artist(a)
a=Arrow3D([o[0], z0[0]], [o[1], z0[1]], [o[2], z0[2]], **arrow_prop_dict)
ax.add_artist(a)
ax.set_xlim([o[0], max_val])
ax.set_ylim([o[0], max_val])
ax.set_zlim([o[0], max_val])

ax.plot([o[0], x1[0]], [o[1], y1[1]], [o[2], z1[2]], color='r', linewidth=1)
ax.plot(x1[0], y1[1], z1[2], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)
ax.plot([x1[0], x1[0]], [y1[1], y1[1]], [o[0], z1[2]], 'k--', linewidth=1)
ax.plot([o[0], x1[0]], [o[0], y1[1]], [o[0], o[0]], 'k--', linewidth=1)

# mark z0 rotation angles (psi)
arc = np.linspace(0,psi)
p = np.array([np.cos(arc),np.sin(arc),arc * 0]) * 0.6
ax.plot(p[0,:],p[1,:],p[2,:],'k')

# mark y1 rotation angles (theta)
arc = np.linspace(0,theta)
p = np.array([np.sin(arc),arc * 0,np.cos(arc)]) * 0.6
p = Rz(psi).dot(p)
ax.plot(p[0,:],p[1,:],p[2,:],'k')

ax.text(1.2*x0[0],1.2*x0[1],1.2*x0[2],r'$x$', **text_options)
ax.text(1.1*y0[0],1.1*y0[1],1.1*y0[2],r'$y$', **text_options)
ax.text(1.1*z0[0],1.1*z0[1],1.1*z0[2],r'$z$', **text_options)
ax.text(0.0,0.0,-0.05,r'$o$', **text_options)

# add psi angle labels
m = 0.55 * ((x0 + x1) / 2.0)
ax.text(m[0], m[1], m[2]-.1, r'$\varphi$', **text_options)

# add theta angle lables
m = 0.9*((x1 - x2))
ax.text(m[0], m[1], m[2], r'$\theta$', **text_options)
ax.text(x2[0]-.1, x2[1]-.1, x2[2]*-1, r'$r$', **text_options)
ax.text(x0[0]-.05, y0[1]-.05, z0[2]+.05, r'$(r,\theta,\varphi)$', **text_options)

ax.set_title('Spherical coordinate system', fontdict={'fontsize': 18, 'fontweight': 'bold'})

# show figure
ax.view_init(elev=10, azim=22)
ax.dist = 10
ax.set_axis_off()
ax.grid(True)

plt.tight_layout()

#%%

plt.savefig(os.path.join(out_path,"spherical_coordinates.svg"), transparent=True)
plt.savefig(os.path.join(out_path,"spherical_coordinates.png"), transparent=True, dpi=450)
plt.savefig(os.path.join(out_path,"spherical_coordinates_white.png"), transparent=False, dpi=450)
plt.close()


#%% Figure 2.5: Leksell frame CT fiducials

frame_fname=r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/axial_frame.png'

fig = plt.figure(figsize=(14,10))
frame_img=plt.imread(frame_fname)
ax = fig.add_subplot(111)
ax.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
ax.axis('off')

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.24,.2))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(140,360),axins.set_ylim(830,1050)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=1, loc2=4,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.24,.4))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(140,360),axins.set_ylim(1760, 1980)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=1, loc2=4,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.24,.6))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(140,360),axins.set_ylim(2900,3110)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=1, loc2=3,linewidth=1, fc="none", ec="0.5" )

ax.annotate('9', xy=(.16, .19), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('8', xy=(.16, .39), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('7', xy=(.16, .59), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')



axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.78,.2))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(3440,3660),axins.set_ylim(830,1050)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=2, loc2=3,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.78,.4))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(3440,3660),axins.set_ylim(1760, 1980)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=2, loc2=3,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.78,.6))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(3440,3660),axins.set_ylim(2900,3110)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=2, loc2=4,linewidth=1, fc="none", ec="0.5" )

ax.annotate('1', xy=(.84, .19), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('2', xy=(.84, .39), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('3', xy=(.84, .59), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')



axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.35,.8))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(770, 990),axins.set_ylim(3830, 4050)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=1, loc2=3,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.54,.8))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(1720, 1940),axins.set_ylim(3830, 4050)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=2, loc2=4,linewidth=1, fc="none", ec="0.5" )

axins = zoomed_inset_axes(ax, 4, loc='center', bbox_transform=ax.figure.transFigure, bbox_to_anchor=(.68,.8))
axins.imshow(np.fliplr(np.rot90(frame_img,2)), cmap='gray',alpha=1, origin="lower")
axins.set_xlim(2830, 3050),axins.set_ylim(3830, 4050)
axins.set_xticks([]),axins.set_yticks([])
mark_inset(ax, axins, loc1=2, loc2=4,linewidth=1, fc="none", ec="0.5" )

ax.annotate('6', xy=(.34, .88), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('5', xy=(.53, .88), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')
ax.annotate('4', xy=(.67, .88), xycoords=ax.figure.transFigure, fontsize= 22, fontweight= 'bold',color='red')


fig.subplots_adjust(top=.7)

plt.suptitle('Leksell frame model G CT fiducials', y=.98,fontproperties=fontTitle)


#%%

plt.savefig(os.path.join(out_path,f"leksell_frame_ct_fiducials.svg"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,f"leksell_frame_ct_fiducials.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,f"leksell_frame_ct_fiducials_white.png"),transparent=False,dpi=450)
plt.close()


#%% Figure 2.10: Iterative closest point registration

input_target=r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/frame-leksell_acq-clinical_run-01.fcsv'
input_fcsv_before=r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/space-leksellg_desc-fiducials_fids_reverse.fcsv'

df_before = pd.read_csv( input_fcsv_before, skiprows=2, usecols=['x','y','z'] )
df_before['x'] = (-1 * df_before['x']) # flip orientation in x
df_before['y'] = (-1 * df_before['y']) # flip orientation in y

df_t = pd.read_csv(input_target, skiprows=2, usecols=['x','y','z'] )
df_t['x'] = -1 * df_t['x'] # flip orientation in x
df_t['y'] = -1 * df_t['y'] # flip orientation in y


tickLocs_x = np.linspace(-100, 100, 9)
tickLabels_x=[str(int(x)+100) for x in tickLocs_x][::-1]
tickLocs_y = np.linspace(-100, 100, 9)
tickLabels_y=[str(int(x)+100) for x in tickLocs_y]
tickLocs_z = np.linspace(-60, 60, 7)
tickLabels_z=[str(int(x)+100) for x in tickLocs_z][::-1]

x = [-95,-95,-95,-95,60,60,60,-60,-60,95,95,95,95]
y = [60,60,-60,-60,115,115,115,115,115,60,60,-60,-60]
z = [60,-60,60,-60,60,-60,60,-60,60,60,-60,60,-60]


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(121, projection = '3d')

ax.set_xlabel('X axis: M-L (mm)', fontsize=14, fontweight='bold', labelpad=16)
ax.set_ylabel('Y axis: A-P (mm)', fontsize=14, fontweight='bold', labelpad=12)
ax.set_zlabel('Z axis: I-S (mm)', fontsize=14, fontweight='bold', labelpad=4)
ax.scatter(df_before['x'][::2], df_before['y'][::2], df_before['z'][::2],s=4, color='#7ff658',alpha=0.7)
ax.plot(x[:4], y[:4], z[:4],linewidth=2, color='#e41a1c')
ax.plot(x[4:9], y[4:9], z[4:9],linewidth=2, color='#e41a1c')
ax.plot(x[9:], y[9:], z[9:],linewidth=2, color='#e41a1c')
ax.text2D(-.15,1, 'a)', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.view_init(elev=25, azim=45)
ax.dist = 10

ax.set_xticks(tickLocs_x, minor=False),ax.set_xticklabels(tickLabels_x)
ax.set_xlim(int(tickLocs_x[0]-20),int(tickLocs_x[-1]+20))
ax.set_yticks(tickLocs_y, minor=False),ax.set_yticklabels(tickLabels_y)
ax.set_ylim(int(tickLocs_y[0]+40),int(tickLocs_y[-1]+20))
ax.set_zticks(tickLocs_z, minor=False)
ax.set_zticklabels(tickLabels_z)
ax.set_zlim(int(tickLocs_z[0]-5),int(tickLocs_z[-1]+5))

ax.w_xaxis.set_pane_color((0, 0, 0, .8))
ax.w_yaxis.set_pane_color((0, 0, 0, .8))
ax.w_zaxis.set_pane_color((0, 0, 0, .8))
ax.xaxis._axinfo["grid"]['color'] = "#FFFFFF60"
ax.yaxis._axinfo["grid"]['color'] = "#FFFFFF60"
ax.zaxis._axinfo["grid"]['color'] = "#FFFFFF60"


ax.scatter(100, -100, 100,facecolors='#17becf',s=40)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', color='k', shrinkA=0, shrinkB=0)
a = Arrow3D([100,60],[-100,-100],[100,100], lw=2, **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([100,100],[-100,-65],[100,100], lw=2, **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([100,100],[-100,-100],[100,70], lw=2, **arrow_prop_dict)
ax.add_artist(a)
ax.text2D(-.24,.89, 'Origin (0,0,0)', fontsize=12, fontweight='bold', transform=ax.transAxes)


ax = fig.add_subplot(122, projection = '3d')
ax.set_xlabel('X axis: M-L (mm)', fontsize=14, fontweight='bold', labelpad=16)
ax.set_ylabel('Y axis: A-P (mm)', fontsize=14, fontweight='bold', labelpad=12)
ax.set_zlabel('Z axis: I-S (mm)', fontsize=14, fontweight='bold', labelpad=4)
ax.scatter(df_t['x'][::4], df_t['y'][::4], df_t['z'][::4], s=4, color='#7ff658', label='Source',alpha=0.7)
ax.plot(x[:4], y[:4], z[:4],linewidth=2, color='#e41a1c')
ax.plot(x[4:9], y[4:9], z[4:9],linewidth=2, color='#e41a1c')
ax.plot(x[9:], y[9:], z[9:],linewidth=2, color='#e41a1c', label='Target')
ax.text2D(-.15,1, 'b)', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.view_init(elev=25, azim=45)
ax.dist = 10

ax.set_xticks(tickLocs_x, minor=False),ax.set_xticklabels(tickLabels_x)
ax.set_xlim(int(tickLocs_x[0]-20),int(tickLocs_x[-1]+20))
ax.set_yticks(tickLocs_y, minor=False),ax.set_yticklabels(tickLabels_y)
ax.set_ylim(int(tickLocs_y[0]+40),int(tickLocs_y[-1]+20))
ax.set_zticks(tickLocs_z, minor=False)
ax.set_zticklabels(tickLabels_z)
ax.set_zlim(int(tickLocs_z[0]-5),int(tickLocs_z[-1]+5))

ax.w_xaxis.set_pane_color((0, 0, 0, .8))
ax.w_yaxis.set_pane_color((0, 0, 0, .8))
ax.w_zaxis.set_pane_color((0, 0, 0, .8))
ax.xaxis._axinfo["grid"]['color'] = "#FFFFFF60"
ax.yaxis._axinfo["grid"]['color'] = "#FFFFFF60"
ax.zaxis._axinfo["grid"]['color'] = "#FFFFFF60"

ax.scatter(100, -100, 100,facecolors='#17becf',s=40)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', color='k', shrinkA=0, shrinkB=0)
a = Arrow3D([100,60],[-100,-100],[100,100], lw=2, **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([100,100],[-100,-65],[100,100], lw=2, **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([100,100],[-100,-100],[100,70], lw=2, **arrow_prop_dict)
ax.add_artist(a)
ax.text2D(-.24,.89, 'Origin (0,0,0)', fontsize=12, fontweight='bold', transform=ax.transAxes)


lgnd=ax.legend(prop={'size': 14}, bbox_to_anchor= (1.4, .7))

#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [40]
lgnd.legendHandles[0]._alpha=1
lgnd.legendHandles[1]._sizes = [40]
lgnd.legendHandles[1]._alpha=1

fig.suptitle('Iterative Closest Point Registration', fontsize=22, fontweight= 'bold')
fig.subplots_adjust(right=0.85)


#%%

plt.savefig(os.path.join(out_path,"iterative_closest_point.svg"),transparent=True)
plt.savefig(os.path.join(out_path,"iterative_closest_point.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,"iterative_closest_point_white.png"),transparent=False,dpi=450)
plt.close()

#%% Figure 3.2: Structuring elements plot

def explode(data):
	shape_arr = np.array(data.shape)
	size = shape_arr[:3]*2 - 1
	exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
	exploded[::2, ::2, ::2] = data
	return exploded

def expand_coordinates(indices):
	x, y, z = indices
	x[1::2, :, :] += 1
	y[:, 1::2, :] += 1
	z[:, :, 1::2] += 1
	return x, y, z

struc_elements = {
	"2d":{
		"square(3) [3x3]": morphology.square(3),
		"disk(2) [5x5]": morphology.disk(2),
		"disk(3) [7x7]": morphology.disk(3)
	},
	"3d": {
		"cube(3) [3x3x3]": morphology.cube(3),
		"ball(2) [5x5x5]": morphology.ball(2),
		"ball(3) [7x7x7]": morphology.ball(3)
	}
}

plot_transparency=False
fig = plt.figure(figsize=(16, 8))
idx = 1 
for plot_dim in list(struc_elements):
	for title, struc in struc_elements[plot_dim].items():
		if plot_dim=='2d':
			cmap = colors.ListedColormap(['#1f77b430','#ff0000ff'])
			ax = fig.add_subplot(2,3,idx)
			#box = ax.get_position()
			#ax.set_position([box.x0+.02, box.y0, box.width * 0.8 , box.height * 0.8])
			ax.pcolormesh(struc, edgecolors='k', linewidth=.1, cmap=cmap, vmin=0,vmax=1)
			ax.set_title(title, fontdict={'fontsize': 18, 'fontweight': 'bold'},y=1.05)
			ax.set_aspect('equal')
			ax.grid(False)
			for i in range(struc.shape[0]):
				for j in range(struc.shape[1]):
					ax.text(j+.5, i+.5, struc[i, j], ha="center", va="center", color="black")
			ax.set_axis_off()
		else:
			ax = fig.add_subplot(2,3,idx, projection='3d')
			if plot_transparency:
				filled = explode(np.ones(struc.shape))
				facecolor = np.array([[['#1f77b410']*filled.shape[0]]*filled.shape[0]]*filled.shape[0])
				facecolor[explode(struc)==1] = '#ff0000ff'
				x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
				ax.voxels(x,y,z,filled, facecolors=facecolor, edgecolors='black', linewidth=.1, shade=False)
			else:
				ax.voxels(struc, facecolors='#ff0000ff', edgecolors='black', linewidth=.5, shade=False)
			ax.set_title(title, fontdict={'fontsize': 18, 'fontweight': 'bold'})
			ax.set_axis_off()
		idx+=1

plt.tight_layout()


#%%

plt.savefig(os.path.join(out_path,f"morphology_kernels.svg"),transparent=True)
plt.savefig(os.path.join(out_path,f"morphology_kernels.png"),transparent=True,dpi=350)
plt.savefig(os.path.join(out_path,f"morphology_kernels_white.png"),dpi=350)

plt.close()
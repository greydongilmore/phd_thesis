#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 18:28:44 2021

@author: greydon
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import shapiro
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter
import glob
from pytablewriter import MarkdownTableWriter
import re
import ast
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.stats.multicomp as multi
import pingouin as pg
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator


def icc(Y, icc_type='ICC(2,1)'):
	''' Calculate intraclass correlation coefficient
	ICC Formulas are based on:
	Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
	assessing rater reliability. Psychological bulletin, 86(2), 420.
	icc1:  x_ij = mu + beta_j + w_ij
	icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
	Code modifed from nipype algorithms.icc
	https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
	Args:
		Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
			measures in columns
		icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)) 
	Returns:
		ICC: (np.array) intraclass correlation coefficient
	'''

	[n, k] = Y.shape

	# Degrees of Freedom
	dfc = k - 1
	dfe = (n - 1) * (k-1)
	dfr = n - 1

	# Sum Square Total
	mean_Y = np.mean(Y)
	SST = ((Y - mean_Y) ** 2).sum()

	# create the design matrix for the different levels
	x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
	x0 = np.tile(np.eye(n), (k, 1))  # subjects
	X = np.hstack([x, x0])

	# Sum Square Error
	predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
								X.T), Y.flatten('F'))
	residuals = Y.flatten('F') - predicted_Y
	SSE = (residuals ** 2).sum()

	MSE = SSE / dfe

	# Sum square column effect - between colums
	SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
	MSC = SSC / dfc  # / n (without n in SPSS results)

	# Sum Square subject effect - between rows/subjects
	SSR = SST - SSC - SSE
	MSR = SSR / dfr
	
	print(MSR-MSE)
	print((MSR + (k-1) * MSE + k * (MSC - MSE) / n))
	if icc_type == 'ICC(2,1)' or icc_type == 'ICC(2,k)':
		# ICC(2,1) = (mean square subject - mean square error) /
		# (mean square subject + (k-1)*mean square error +
		# k*(mean square columns - mean square error)/n)
		if icc_type == 'ICC(2,k)':
			k = 1
		ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

	elif icc_type == 'ICC(3,1)' or icc_type == 'ICC(3,k)':
		# ICC(3,1) = (mean square subject - mean square error) /
		# (mean square subject + (k-1)*mean square error)
		if icc_type == 'ICC(3,k)':
			k = 1
		ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

	return ICC

import numpy as np
import scipy.io as sio
import scipy.stats as sstats

def ICC_rep_anova(Y):
	'''
	the data Y are entered as a 'table' ie subjects are in rows and repeated
	measures in columns
	One Sample Repeated measure ANOVA
	Y = XB + E with X = [FaTor / Subjects]
	'''

	[nb_subjects, nb_conditions] = Y.shape
	dfc = nb_conditions - 1
	dfe = (nb_subjects - 1) * dfc
	dfr = nb_subjects - 1

	# Compute the repeated measure effect
	# ------------------------------------

	# Sum Square Total
	mean_Y = np.mean(Y)
	SST = ((Y - mean_Y)**2).sum()

	# create the design matrix for the different levels
	x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
	x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
	X = np.hstack([x, x0])

	# Sum Square Error
	predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
	residuals = Y.flatten('F') - predicted_Y
	SSE = (residuals**2).sum()

	residuals.shape = Y.shape

	MSE = SSE / dfe

	# Sum square session effect - between colums/sessions
	SSC = ((np.mean(Y, 0) - mean_Y)**2).sum() * nb_subjects
	MSC = SSC / dfc / nb_subjects

	session_effect_F = MSC / MSE

	# Sum Square subject effect - between rows/subjects
	SSR = SST - SSC - SSE
	MSR = SSR / dfr

	# ICC(3,1) = (mean square subjeT - mean square error) /
	#            (mean square subjeT + (k-1)*-mean square error)
	ICC = (MSR - MSE) / (MSR + dfc * MSE)

	e_var = MSE  # variance of error
	r_var = (MSR - MSE) / nb_conditions  # variance between subjects

	return ICC, r_var, e_var, session_effect_F, dfc, dfe

def sorted_nicely(data, reverse = False):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	
	return sorted(data, key = alphanum_key, reverse=reverse)


def outlier_removal_IQR(data):
	Q1=data.quantile(0.25)
	Q3=data.quantile(0.75)
	iqr=6*(Q3-Q1)
	q1_idx = data[data < Q1-iqr]
	data = data.drop(q1_idx)
	q3_idx = data[data > Q3+iqr]
	data = data.drop(q3_idx)
	
	return data

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

title_dic = {
	'matchCentroids': 'Match centroids',
	'numLandmarks': 'Num landmarks',
	'maximummeandistance': 'Max mean distance',
}


#%%

input_dir=r'/media/veracrypt6/projects/stealthMRI/derivatives/validation/version_02'

#patient_ignore=['sub-P084','sub-P062','sub-P156','sub-P161']
patient_ignore=[]
out_path='/media/greydon/KINGSTON/phdCandidacy/thesis/imgs'
xls = pd.ExcelFile('/media/veracrypt6/projects/stealthMRI/resources/excelFiles/dbsChartReview.xlsx')

#--- Import excel data and clean ---
raw_data = xls.parse('Sheet14', header=0)
df = raw_data[raw_data["stealth_error"].notna()]
df.round(2);


data_dir=input_dir+'/derivatives'


avg_cluster=[]
for isub in sorted_nicely([x for x in os.listdir(data_dir) if x not in patient_ignore]):
	subject=int(''.join([x for x in isub if x.isnumeric()]))
	for ifile in glob.glob(data_dir+f'/{isub}/frame/*_fids.tsv'):
		df_tmp = pd.read_csv( ifile, header=0,sep='\t')
		if 'centroids' in os.path.basename(ifile):
			for icluster in df_tmp['label'].unique():
				cluster_data=df_tmp[df_tmp['label']==icluster]
				avg_cluster.append([subject, icluster,np.mean(cluster_data['dist_x']),np.mean(cluster_data['dist_y']),np.mean(cluster_data['dist_z']),np.mean(cluster_data['error'])])

avg_cluster=pd.DataFrame(avg_cluster, columns = ['subject','label','x','y','z','error'])

tg_error=[]
for isub in df['subject']:
	tg_error.append(avg_cluster[avg_cluster['subject']==isub]['error'].mean())

error_data=np.c_[df.loc[:,"subject"].values.astype(int), df.loc[:,"stealth_error"].values,tg_error]
error_data=pd.DataFrame(error_data)
error_data.rename(columns={1: 'stealth_error', 2: 'trajectoryGuide_error',0:'subject'}, inplace=True)
error_data = error_data[error_data["trajectoryGuide_error"].notna()].reset_index(drop=True)

avg_cluster_coord=[]
for isub in sorted_nicely([x for x in os.listdir(data_dir) if x not in patient_ignore]):
	subject=int(''.join([x for x in isub if x.isnumeric()]))
	for ifile in glob.glob(data_dir+f'/{isub}/frame/*_fids.tsv'):
		df_tmp = pd.read_csv( ifile, header=0,sep='\t')
		if 'clusters' in os.path.basename(ifile):
			for icluster in df_tmp['label'].unique():
				df_tmp['subject'] = np.repeat(subject, df_tmp.shape[0])
				avg_cluster_coord.append(df_tmp)
				
avg_cluster_coord = pd.concat(avg_cluster_coord)
avg_cluster_coord=pd.DataFrame(avg_cluster, columns = ['subject','label','x','y','z','error'])


#%%



data_dir=input_dir+'/derivatives'
avg_cluster_params_seg=[]
avg_cluster_coord_params_seg=[]

for isub in sorted_nicely([x for x in os.listdir(data_dir) if x not in patient_ignore])[:-1]:
	subject=int(''.join([x for x in isub if x.isnumeric()]))
	for ifile in glob.glob(data_dir+f'/{isub}/frame/*_fids.tsv'):
		params=os.path.basename(ifile).split('_')[2:-2]
		if params:
			param_vals=[]
			params_labels=[]
			for iparam in params:
				if iparam.split('-')[1]== 'True':
					param_vals.append(1)
				elif iparam.split('-')[1]== 'False':
					param_vals.append(0)
				else:
					param_vals.append(ast.literal_eval(iparam.split('-')[1]))
				
				params_labels.append(iparam.split('-')[0])
			
			df = pd.read_csv( ifile, header=0,sep='\t')
			if 'centroids' in os.path.basename(ifile):
				for icluster in df['label'].unique():
					cluster_data=df[df['label']==icluster]
					avg_cluster_params_seg.append([isub]+param_vals +[icluster,np.mean(cluster_data['dist_x']),np.mean(cluster_data['dist_y']),np.mean(cluster_data['dist_z']),np.mean(cluster_data['error'])])
					avg_cluster_coord_params_seg.append([isub]+param_vals +[icluster,'x',np.mean(cluster_data['dist_x'])])
					avg_cluster_coord_params_seg.append([isub]+param_vals +[icluster,'y',np.mean(cluster_data['dist_y'])])


avg_cluster_coord_params_seg=pd.DataFrame(avg_cluster_coord_params_seg,columns = ['subject']+params_labels+['label','coord','error'])
avg_cluster_params_seg=pd.DataFrame(avg_cluster_params_seg,columns = ['subject']+params_labels+['label','x','y','z','error'])


data_dir=input_dir+'/frame_segmentation_tuning_intensity_weight'
avg_cluster_params_seg_iw=[]
avg_cluster_coord_params_seg_iw=[]

for isub in sorted_nicely([x for x in os.listdir(data_dir) if x not in patient_ignore])[:-1]:
	subject=int(''.join([x for x in isub if x.isnumeric()]))
	for ifile in glob.glob(data_dir+f'/{isub}/frame/*_fids.tsv'):
		params=os.path.basename(ifile).split('_')[2:-2]
		if params:
			param_vals=[]
			params_labels=[]
			for iparam in params:
				if iparam.split('-')[1]== 'True':
					param_vals.append(1)
				elif iparam.split('-')[1]== 'False':
					param_vals.append(0)
				else:
					param_vals.append(ast.literal_eval(iparam.split('-')[1]))
				
				params_labels.append(iparam.split('-')[0])
			
			df = pd.read_csv( ifile, header=0,sep='\t')
			if 'centroids' in os.path.basename(ifile):
				for icluster in df['label'].unique():
					cluster_data=df[df['label']==icluster]
					avg_cluster_params_seg_iw.append([isub]+param_vals +[icluster,np.mean(cluster_data['dist_x']),np.mean(cluster_data['dist_y']),np.mean(cluster_data['dist_z']),np.mean(cluster_data['error'])])
					avg_cluster_coord_params_seg_iw.append([isub]+param_vals +[icluster,'x',np.mean(cluster_data['dist_x'])])
					avg_cluster_coord_params_seg_iw.append([isub]+param_vals +[icluster,'y',np.mean(cluster_data['dist_y'])])
# 			else:
# 				for iclust in np.where(np.diff(df['z']) > .6):
# 					cluster_data=df.iloc[:iclust,:]
# 					sum_pixel_values = sum(cluster_data['intensity'])
# 					x_iw = sum(cluster_data['x'].to_numpy() * cluster_data['intensity'].to_numpy()) / sum_pixel_values
# 					y_iw = sum(cluster_data['y'].to_numpy() * cluster_data['intensity'].to_numpy()) / sum_pixel_values
# 					z_iw = sum(cluster_data['z'].to_numpy() * cluster_data['intensity'].to_numpy()) / sum_pixel_values
# 					
# 				for islice in np.unique([round(x,1) for x in df['z']]):
# 					cluster_data=df[df['label']==icluster]
# 					avg_cluster_params_seg.append([isub]+param_vals +[icluster,np.mean(cluster_data['dist_x']),np.mean(cluster_data['dist_y']),np.mean(cluster_data['dist_z']),np.mean(cluster_data['error'])])
# 					avg_cluster_coord_params_seg.append([isub]+param_vals +[icluster,'x',np.mean(cluster_data['dist_x'])])
# 					avg_cluster_coord_params_seg.append([isub]+param_vals +[icluster,'y',np.mean(cluster_data['dist_y'])])

avg_cluster_coord_params_seg_iw=pd.DataFrame(avg_cluster_coord_params_seg_iw,columns = ['subject']+params_labels+['label','coord','error'])
avg_cluster_params_seg_iw=pd.DataFrame(avg_cluster_params_seg_iw,columns = ['subject']+params_labels+['label','x','y','z','error'])






data_dir=input_dir+'/frame_detection_tuning'
avg_cluster_params=[]
avg_cluster_coord_params=[]

for isub in sorted_nicely([x for x in os.listdir(data_dir) if x not in patient_ignore])[:-1]:
	subject=int(''.join([x for x in isub if x.isnumeric()]))
	for ifile in glob.glob(data_dir+f'/{isub}/frame/*centroids_fids.tsv'):
		params=os.path.basename(ifile).split('_')[2:-2]
		if params:
			param_vals=[]
			params_labels=[]
			for iparam in params:
				if iparam.split('-')[1]== 'True':
					param_vals.append(1)
				elif iparam.split('-')[1]== 'False':
					param_vals.append(0)
				else:
					param_vals.append(ast.literal_eval(iparam.split('-')[1]))
				
				params_labels.append(iparam.split('-')[0])
			
			df = pd.read_csv( ifile, header=0,sep='\t')
			for icluster in df['label'].unique():
				cluster_data=df[df['label']==icluster]
				avg_cluster_params.append([isub]+param_vals +[icluster,np.mean(cluster_data['dist_x']),np.mean(cluster_data['dist_y']),np.mean(cluster_data['dist_z']),np.mean(cluster_data['error'])])
				avg_cluster_coord_params.append([isub]+param_vals +[icluster,'x',np.mean(cluster_data['dist_x'])])
				avg_cluster_coord_params.append([isub]+param_vals +[icluster,'y',np.mean(cluster_data['dist_y'])])


avg_cluster_coord_params=pd.DataFrame(avg_cluster_coord_params,columns = ['subject']+params_labels+['label','coord','error'])
avg_cluster_params=pd.DataFrame(avg_cluster_params,columns = ['subject']+params_labels+['label','x','y','z','error'])


#%%

avg_cluster[['error','maximummeandistance']].groupby('threshold').mean()
avg_cluster_params[['error','maximummeandistance']].groupby(['maximummeandistance']).mean()

avg_cluster[['label','x','y','z']].groupby('label').mean()
avg_cluster[['label','error','threshold']].groupby(['label','threshold'], as_index=False).mean()

df=pd.DataFrame(np.c_[avg_cluster['label'].unique(),
					  [N +P for N,P in zip([f'{x[0]:.3f}' for x in avg_cluster[['label','x']].groupby('label').mean().values ],[f' ({x[0]:.3f})' for x in avg_cluster[['label','x']].groupby('label').std().values])],
					  [N +P for N,P in zip([f'{x[0]:.3f}' for x in avg_cluster[['label','y']].groupby('label').mean().values ],[f' ({x[0]:.3f})' for x in avg_cluster[['label','y']].groupby('label').std().values])],
					  [N +P for N,P in zip([f'{x[0]:.3f}' for x in avg_cluster[['label','z']].groupby('label').mean().values ],[f' ({x[0]:.3f})' for x in avg_cluster[['label','z']].groupby('label').std().values])],
					  [f'{x[0]:.3f}' for x in avg_cluster[['label','error']].groupby('label').mean().values]])

print(df.to_csv(header=None, index=None))


sum_df=avg_cluster[['label','error']].groupby(['label'], as_index=False).mean()
std_df=avg_cluster[['label','error']].groupby(['label'], as_index=False).std()

values_matrix=[]
for ilabel in sum_df['label'].unique():
	value_temp=np.array([ilabel])
	for ithres in sum_df['threshold'].unique():
		error=sum_df[(sum_df['threshold']==ithres) & (sum_df['label']==ilabel)]['error'].to_numpy()
		error_std=std_df[(std_df['threshold']==ithres) & (std_df['label']==ilabel)]['error'].to_numpy()
		
		value_temp = np.c_[value_temp, [f'{error[0]:.3f} ' + u"\u00B1" + f' {error_std[0]:.3f}']]
	values_matrix.append(value_temp[0])

avg_error=avg_cluster[['error','threshold']].groupby('threshold', as_index=False).mean()
std_error=avg_cluster[['error','threshold']].groupby('threshold', as_index=False).std()

value_temp=np.array(['overall'])
for ithres in avg_error['threshold'].unique():
	error=avg_error[avg_error['threshold']==ithres]['error'].to_numpy()
	error_std=std_error[std_error['threshold']==ithres]['error'].to_numpy()
	value_temp = np.c_[value_temp,[f'{error[0]:.3f} ' + u"\u00B1" + f' {error_std[0]:.3f}']]
values_matrix.append(value_temp[0])

values_matrix=np.stack(values_matrix).tolist()

writer = MarkdownTableWriter(
	table_name="example_table",
	headers=["label"]+[str(x) for x in avg_error['threshold'].unique()],
	value_matrix=values_matrix
	)

writer.write_table()

#%%


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

sns.lineplot(x = 'minthreshold', y = "error", data=avg_cluster_params_seg, ax=ax, linewidth=2, ci=None, marker='o',label='Mean')
sns.lineplot(x = 'minthreshold', y = "error", data=avg_cluster_params_seg_iw, ax=ax, linewidth=2, ci=None, marker='o',label='Intensity-weighted')
ax.set_xlabel('Threshold (HU)', fontweight='bold',fontsize=18,labelpad=12)
ax.set_ylabel('Regisration Error (mm)', fontweight='bold',fontsize=18,labelpad=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title("Leksell frame CT segmentation pixel intensity\nthreshold tuning curve", y=1.05,fontsize= 22, fontweight='bold')

tickLocs_x=sorted(avg_cluster_params_seg['minthreshold'].unique())
cadenceX= tickLocs_x[2] - tickLocs_x[1]
ax.set_xlim(tickLocs_x[0]-cadenceX,tickLocs_x[-1]+cadenceX)
tickLabels=[str(x) for x in tickLocs_x]
ax.set_xticks(tickLocs_x, minor=False),ax.set_xticklabels(tickLabels)

tickLocs_y=ax.yaxis.get_ticklocs()
ax.set_ylim(tickLocs_y[0],tickLocs_y[-1])
tickLabels=[format(float(x), '.3f') for x in tickLocs_y]
ax.set_yticks(tickLocs_y, minor=False),ax.set_yticklabels(tickLabels)

lgnd=ax.legend(prop={'size': 12}, bbox_to_anchor= (1.01, .7))

plt.tight_layout()


#%%


plt.savefig(os.path.join(out_path,"segmentation_parameter_tuning.svg"),transparent=True)
plt.savefig(os.path.join(out_path,"segmentation_parameter_tuning.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,"segmentation_parameter_tuning_white.png"),transparent=False,dpi=450)
plt.close()


#%%


plot_dat=avg_cluster_params[(avg_cluster_params['maximummeandistance'] !=0.1) & (avg_cluster_params['numLandmarks'] >=200)][['error']+params_labels].copy()
params_clean = [x for x in params_labels if not any(y ==x for y in ('matchCentroids','maximummeandistance'))]
if 'maximummeandistance' in list(plot_dat):
	plot_dat['maximummeandistance']=['{:f}'.format(x).rstrip('0') for x in plot_dat['maximummeandistance']]

plot_dat['numLandmarks']=[str(x) for x in plot_dat['numLandmarks']]


fig, axes = plt.subplots(nrows=len(params_clean)*2, figsize=(14,10))
letter_label=[chr(i) for i in range(ord('a'),ord('z')+1)]
for iparam in range(len(params_clean)):
	match_cnt=1
	ylims=[]
	for imatch in plot_dat['matchCentroids'].unique():
		#ax = fig.add_subplot(len(params_clean)*2, 1, iparam+match_cnt,sharex = ax)
		for idist in plot_dat['maximummeandistance'].unique()[1:]:
			if imatch ==0:
				label=f"dist = {idist}"
				sub_title="No centroid matching"
			else:
				label=f"dist = {idist}"
				sub_title="Centroid matching"
			plt_tmp=plot_dat[(plot_dat['matchCentroids']==imatch) & (plot_dat['maximummeandistance']==idist)]
			plt_tmp=plt_tmp.groupby(params_clean[iparam],as_index=False).mean()
			sns.lineplot(x = params_clean[iparam], y = "error", data=plt_tmp,ax=axes[imatch], linewidth=1,ci=None, label=label, marker='o')
		axes[imatch].set_xlabel(title_dic[params_clean[iparam]], fontweight='bold',fontsize=18,labelpad=12)
		axes[imatch].set_ylabel('Regisration Error (mm)', fontweight='bold',fontsize=18,labelpad=18)
		axes[imatch].set_title(sub_title,fontsize=20,y=1.1)
		axes[imatch].spines['right'].set_visible(False)
		axes[imatch].spines['top'].set_visible(False)
		axes[imatch].tick_params(axis='both', which='major', labelsize=14)
		axes[imatch].legend(bbox_to_anchor= (1.2, .8),fontsize = 12)
		axes[imatch].text(-.05, 1.1,f'{letter_label[match_cnt-1]})', transform=axes[imatch].transAxes, fontsize=18, fontweight='bold')
		ylims.append(axes[imatch].yaxis.get_ticklocs())
		
		match_cnt+=1
		
	ylim_min=round(min([min(a)for a in ylims]),4)
	ylim_max=round(max([max(a)for a in ylims]),4)
	for imatch in plot_dat['matchCentroids'].unique():
		tickLocs_y=axes[imatch].yaxis.get_ticklocs()
		cadenceY= tickLocs_y[2] - tickLocs_y[1]
		tickLocs_y = np.arange(ylim_min, ylim_max, .0001)
		axes[imatch].set_ylim(ylim_min,ylim_max)
		tickLabels=[format(float(x), '.4f') for x in tickLocs_y]
		axes[imatch].set_yticks(tickLocs_y, minor=False),axes[imatch].set_yticklabels(tickLabels)

plt.tight_layout(pad=3)
fig.subplots_adjust(right=0.85,top=.85)
fig.suptitle(f"{title_dic[params_clean[iparam]]} and centroid matching\ntuning curves", y=1,x=.55, fontsize= 24, fontweight='bold')



#%%

plt.savefig(os.path.join(out_path,"icp_parameter_tuning.svg"),transparent=True)
plt.savefig(os.path.join(out_path,"icp_parameter_tuning.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,"icp_parameter_tuning_white.png"),transparent=False,dpi=450)
plt.close()

#%% descriptive plots

error_data_melt=np.c_[np.r_[error_data.loc[:,["stealth_error",'subject']].values,error_data.loc[:,["trajectoryGuide_error",'subject']].values],
					  np.r_[np.repeat(1,len(error_data.loc[:,"stealth_error"].values)).astype(int), np.repeat(2,len(error_data.loc[:,"trajectoryGuide_error"].values)).astype(int)]]

error_data_melt=pd.DataFrame(error_data_melt)
error_data_melt.rename(columns={0: 'error', 1: 'subject',2:'system'}, inplace=True)

error_data_melt.loc[error_data_melt.system==1, 'system'] = "StealthStation"
error_data_melt.loc[error_data_melt.system==2, 'system'] = "trajectoryGuide"

fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(4, 4)

# violin plot of distributions
ax = fig.add_subplot(gs[:2, :2])
ax = sns.violinplot(x="system", y="error",data=error_data_melt,
					split=True, inner="quartile", color="0.8", width=.5)

sns.stripplot(x="system", y="error", data=error_data_melt, jitter=True, ax=ax)
ax.set_xticks([0,1])
ax.set_xticklabels(['StealthStation','trajectoryGuide'])
#ax.set_xticklabels(['StealthStation','trajectoryGuide'], fontsize=14)
ax.set_xlabel('Navigation System', fontweight='bold',fontsize=18,labelpad=12)
ax.set_ylabel('Registration Error (mm)', fontweight='bold',fontsize=18,labelpad=18)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('Distribution of FRE', y=1.05, fontsize= 20)
ax.patch.set_alpha(0)
ax.set_xlim(-0.5,1.5)
ax.text(-.18, 1,'a)', transform=ax.transAxes, fontsize=18, fontweight='bold')


pvalues = [stats.ttest_rel(error_data["stealth_error"], error_data["trajectoryGuide_error"], alternative="two-sided").pvalue]
annotator = Annotator(ax, [('StealthStation','trajectoryGuide')], data=error_data_melt, x="system", y="error")
annotator.configure(text_format="simple", line_height=0.02, line_width=2, line_offset=3)
annotator.set_pvalues(pvalues)


# statistical annotation
x1, x2 = 0, 1	# columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = error_data_melt['error'].max() + .16, .02, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, annotator.get_annotations_text()[0], ha='center', va='bottom', color=col, fontsize=12)



# histogram plot
mean1, mean2 = np.mean(error_data["stealth_error"]), np.mean(error_data["trajectoryGuide_error"])
std1, std2 = np.std(error_data["stealth_error"], ddof=1), np.std(error_data["trajectoryGuide_error"], ddof=1)
sampling_difference=error_data["stealth_error"]-error_data["trajectoryGuide_error"]

ax = fig.add_subplot(gs[:2, 2:])
sns.histplot(sampling_difference, kde=False, bins=20,color='blue', ax=ax)
plt.title('Distribution of difference in FRE', fontsize=20, y=1.05)
ax.set_xlabel("Difference in FRE", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=18)
ax.set_ylabel("Count", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)

tickLocs = ax.xaxis.get_ticklocs()
cadenceX = tickLocs[2] - tickLocs[1]
tickLocs_x = np.linspace(0,abs(tickLocs.max()),4)[1:]
tickLocs_neg_x = np.array([x*-1 for x in tickLocs_x])
tickLocs_x = np.r_[tickLocs_neg_x[::-1],[0],tickLocs_x]
tickLabels = [f'{x:.2f}' for x in tickLocs_x]
ax.set_xticks(tickLocs_x, minor=False)
ax.set_xticklabels(tickLabels)
ax.set_xlim(tickLocs_x[0],tickLocs_x[-1])

ax.text(-.18, 1,'b)', transform=ax.transAxes, fontsize=18, fontweight='bold')


# q-q plot
ax = fig.add_subplot(gs[2:4, 1:3])
normality_plot, stat = stats.probplot(sampling_difference, plot= sns.mpl.pyplot, dist="norm")
ax.set_title("Q-Q plot of difference in FRE", fontsize= 20, y=1.05)
ax.set_xlabel("Theoretical quantiles", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=18)
ax.set_ylabel("Sample quantiles", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)

stat, p = shapiro(sampling_difference)
ax.text(.5, .1, 'Shapiro-Wilk: W=%.3f, p=%.3f' % (stat, p), fontsize=12,transform = ax.transAxes)
ax.text(-.18, 1,'c)', transform=ax.transAxes, fontsize=18, fontweight='bold')


fig.tight_layout(pad=4)



#%%


plt.savefig(os.path.join(out_path,"frame_error_distribution.svg"),transparent=True)
plt.savefig(os.path.join(out_path,"frame_error_distribution.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,"frame_error_distribution_white.png"),transparent=False,dpi=450)
plt.close()


#%%


slope, intercept, r_value, p_value, std_err = stats.linregress(error_data["stealth_error"], error_data["trajectoryGuide_error"])
label_line_1 = r'$y={0:.1f}x+{1:.1f}'.format(slope,intercept)

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14,7))
icc_fre=icc(np.c_[error_data["stealth_error"],error_data["trajectoryGuide_error"]],'ICC(2,k)')
ax=sns.regplot(x="stealth_error", y="trajectoryGuide_error", data=error_data, fit_reg=False, ax=ax1)
ax1.set_ylim(0, .6),ax1.set_xlim(0, .6)
ax1.set_xlabel("StealthStation FRE", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=10)
ax1.set_ylabel("trajectoryGuide FRE", fontsize=fontAxis['size'], fontweight=fontAxis['weight'],labelpad=10)
ax1.text(.65, .95, f'ICC={icc_fre:.3f}', fontsize=14,transform=ax1.transAxes)
ax1.text(-.18,1,'a)', transform=ax1.transAxes, fontsize=18, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=14)

x0, x1 = ax1.get_xlim()
y0, y1 = ax1.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
ax1.plot(lims, lims, ':k')



right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

percentage=False
limitOfAgreement=1.96
confidenceInterval=0.95

data1=error_data["stealth_error"]
data2=error_data["trajectoryGuide_error"]

mean = np.mean([data1,data2], axis=0)
std=np.std([data1,data2], axis=1)
if percentage:
	diff = ((data1 - data2) / mean) * 100
else:
	diff = data1 - data2

md = np.mean(diff)
sd = np.std(diff, axis=0)-.01
confidenceIntervals_mean = stats.t.interval(confidenceInterval, len(diff)-1, loc=md, scale=sd/np.sqrt(len(diff)))
seLoA = ((1/len(diff)) + (limitOfAgreement**2 / (2 * (len(diff) - 1)))) * (sd**2)
loARange = np.sqrt(seLoA) * stats.t._ppf((1-confidenceInterval)/2., len(diff)-1)
confidenceIntervals_uLoA = ((md + limitOfAgreement*sd) + loARange,(md + limitOfAgreement*sd) - loARange)
confidenceIntervals_lLoA = ((md - limitOfAgreement*sd) + loARange,(md - limitOfAgreement*sd) - loARange)

ax2.axhspan(confidenceIntervals_mean[0], confidenceIntervals_mean[1], facecolor='#6495ED', alpha=0.2)
ax2.axhspan(confidenceIntervals_uLoA[0], confidenceIntervals_uLoA[1], facecolor='coral', alpha=0.2)
ax2.axhspan(confidenceIntervals_lLoA[0], confidenceIntervals_lLoA[1], facecolor='coral', alpha=0.2)

ax2.axhline(md, color='#6495ED', linestyle='--')
ax2.axhline(md + limitOfAgreement*sd, color='coral', linestyle='--')
ax2.axhline(md - limitOfAgreement*sd, color='coral', linestyle='--')
ax2.axhline(0, color='black', linestyle='-')

ax2.scatter(mean, diff, alpha=0.5, c='#6495ED')
trans = transforms.blended_transform_factory(ax2.transAxes, ax2.transData)
limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement*sd)
offset = (limitOfAgreementRange / 100.0) * 1.5
ax2.text(-.2,1,'b)', transform=ax2.transAxes, fontsize=18, fontweight='bold')

ax2.text(0.98, md - (offset*6), 'Mean', ha="right", va="bottom", transform=trans,color='black', fontsize=12)
ax2.text(0.98, md - (offset*6), f'{md:.3f}', ha="right", va="top", transform=trans,color='black', fontsize=12)
ax2.text(0.98, md + (limitOfAgreement * sd) + offset, f'+{limitOfAgreement:.3f} SD', ha="right", va="bottom", transform=trans,color='black', fontsize=12)
ax2.text(0.98, md + (limitOfAgreement * sd) - offset, f'{md + limitOfAgreement*sd:.3f}', ha="right", va="top", transform=trans,color='black', fontsize=12)
ax2.text(0.98, md - (limitOfAgreement * sd) - offset, f'-{limitOfAgreement:.3f} SD', ha="right", va="top", transform=trans,color='black', fontsize=12)
ax2.text(0.98, md - (limitOfAgreement * sd) + offset, f'{md - limitOfAgreement*sd:.3f}', ha="right", va="bottom", transform=trans,color='black', fontsize=12)

if percentage:
	ax2.set_ylabel('Percentage difference (Stealthstation - trajectoryGuide)', fontdict=fontAxis,labelpad=10)
else:
	ax2.set_ylabel('Difference in FRE', fontdict=fontAxis,labelpad=10)
	
ax2.set_xlabel('Mean of FRE', fontdict=fontAxis,labelpad=10)

tickLocs = ax2.xaxis.get_ticklocs()
cadenceX = tickLocs[2] - tickLocs[1]
tickLocs = np.linspace(min(mean)+(cadenceX)*-1,max(mean)+cadenceX,7)
ax2.xaxis.set_major_locator(ticker.FixedLocator(tickLocs))
ax2.set_xlim(min(tickLocs), max(tickLocs))
ax2.spines['bottom'].set_bounds(min(tickLocs), max(tickLocs))


tickLocs = ax2.yaxis.get_ticklocs()
cadenceY = tickLocs[2] - tickLocs[1]
tickLocs = np.arange(cadenceY, max(diff)*3, cadenceY)
tickLocs = np.r_[-tickLocs[::-1], 0, tickLocs]
ax2.yaxis.set_major_locator(ticker.FixedLocator(tickLocs))
ax2.set_ylim(min(tickLocs), max(tickLocs))

ax2.tick_params(axis='both', which='major', labelsize=14)

# Only draw spine between extent of the data
ax2.spines['left'].set_bounds(min(tickLocs), max(tickLocs))
# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.suptitle('Frame Fiducial Registration Error (FRE): Stealthstation vs. trajectoryGuide', fontproperties=fontTitle)

ax.patch.set_alpha(0)
fig.tight_layout(pad=3.0)


#%%


plt.savefig(os.path.join(out_path,"frame_error_bland_altman.svg"),transparent=True)
plt.savefig(os.path.join(out_path,"frame_error_bland_altman.png"),transparent=True,dpi=450)
plt.savefig(os.path.join(out_path,"frame_error_bland_altman_white.png"),transparent=False,dpi=450)
plt.close()


#%%

from scipy.stats import ttest_ind,t

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

error_data.apply(cv)


ind_t_test=ttest_ind(error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]]['error'],error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]]['error'])

N1=error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]].shape[0]
N2=error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]].shape[0]
df = error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]].shape[0] + error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]].shape[0] - 2


data1_mean = error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]]['error'].mean()
data2_mean = error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]]['error'].mean()
diff_mean = data1_mean - data2_mean

data1_std = error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]]['error'].std()
data2_std = error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]]['error'].std()
std_N1N2 = np.sqrt(((N1 - 1)*(data1_std)**2 + (N2 - 1)*(data2_std)**2) / df)
MoE = t.ppf(0.975, df) * std_N1N2 * np.sqrt(1/N1 + 1/N2)

print('The results of the independent t-test are: \n\tt-value = {:4.3f}\n\tp-value = {:4.3f}'.format(ind_t_test[0],ind_t_test[1]))
print ('\nThe difference between groups is {:3.3f} [{:3.3f} to {:3.3f}] (mean [95% CI])'.format(diff_mean, diff_mean - MoE, diff_mean + MoE))


stat,pval=stats.ttest_rel(error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]]['error'], error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]]['error'])
print(stat,pval)

posthoc = pg.pairwise_ttests(data=error_data_melt, dv='error', between='system',parametric=True, padjust='fdr_bh', effsize='hedges')
pg.print_table(posthoc, floatfmt='.3f')
mcDate = multi.MultiComparison(error_data_melt['error'], error_data_melt['system'])
Results = mcDate.tukeyhsd()
print(Results)



#%%


fig, axs = plt.subplots(1,figsize=(12,8))
sns.boxplot(x="label", y="error",data=avg_cluster_coord,ax=axs)

sns.swarmplot(x="label", y="error", data=avg_cluster_coord, ax=axs, size=3,color=".6")
axs.set_xticklabels(avg_cluster_coord['label'].unique(), fontweight='bold',fontsize=14)
axs.set_xlabel('', fontweight='bold',fontsize=18,labelpad=18)
axs.set_xlabel('Frame fiducial number', fontweight='bold',fontsize=18,labelpad=18)
axs.set_ylabel('Eucilidean distance (mm)', fontweight='bold',fontsize=18,labelpad=18)
# Hide the right and top spines
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=14)

arr_img = plt.imread(r'/media/greydon/KINGSTON34/phdCandidacy/thesis/imgs/axial_frame.png', format='png')
newax = fig.add_axes([.81, 0.46, 0.15, 0.15], anchor='NE', zorder=1)
newax.imshow(arr_img)
newax.axis('off')
axs.annotate('', xy=(1.20,.41), xycoords='axes fraction', xytext=(1.05,.41), arrowprops=dict(arrowstyle="<->", color='black',linewidth=2),transform=axs.transAxes)
axs.annotate('', xy=(1.045,.61), xycoords='axes fraction', xytext=(1.045,.42), arrowprops=dict(arrowstyle="<->", color='black',linewidth=2),transform=axs.transAxes)
axs.text(1.09, .38,'X-axis', transform=axs.transAxes, fontsize=12, fontweight='bold')
axs.text(1.02, .47,'Y-axis', transform=axs.transAxes, fontsize=12, fontweight='bold', rotation=90)


#lgnd = ax.legend(fontsize = 15, bbox_to_anchor= (1.1, 1.05), title="Axis", title_fontsize = 18, shadow = True, facecolor = 'white')
plt.suptitle('Source-Target Euclidean distance after registration', y=1,fontproperties=fontTitle)

plt.tight_layout()
plt.subplots_adjust(right=0.82)

#%%

plt.savefig(os.path.join(out_path,f"frame_error_distance_violin.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"frame_error_distance_violin.png"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"frame_error_distance_violin_white.png"),transparent=False,dpi=400)
plt.close()

#%%

def outlier_removal_IQR(data, feature):
	Q1=data[feature].quantile(0.25)
	Q3=data[feature].quantile(0.75)
	iqr=2.5*(Q3-Q1)
	q1_idx = data[feature][data[feature] < Q1-iqr].index
	data = data.drop(q1_idx)
	q3_idx = data[feature][data[feature] > Q3+iqr].index
	data = data.drop(q3_idx)
	
	return data

avg_cluster_cleaned=outlier_removal_IQR(avg_cluster, 'error')


fig, axs = plt.subplots(1,figsize=(12,8))
sns.boxplot(x="label", y="error",data=avg_cluster_cleaned,ax=axs)

sns.swarmplot(x="label", y="error", data=avg_cluster_cleaned, ax=axs, size=3,color=".6")
axs.set_xticklabels(avg_cluster['label'].unique(), fontweight='bold',fontsize=14)
axs.set_xlabel('', fontweight='bold',fontsize=18,labelpad=18)
axs.set_xlabel('Frame fiducial number', fontweight='bold',fontsize=18,labelpad=18)
axs.set_ylabel('Distance from target (mm)', fontweight='bold',fontsize=18,labelpad=18)
# Hide the right and top spines
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.tick_params(axis='both', which='major', labelsize=14)


#lgnd = ax.legend(fontsize = 15, bbox_to_anchor= (1.1, 1.05), title="Axis", title_fontsize = 18, shadow = True, facecolor = 'white')
plt.suptitle('Source Fiducial Distance From Target After Registration', y=1,fontproperties=fontTitle)

plt.tight_layout()
plt.subplots_adjust(right=0.82)


#%% Grouped violin plot


fig, axs = plt.subplots(2,sharex=True,figsize=(12,10))
sns.violinplot(x="label", y="x",data=avg_cluster, inner="quartile", color=".6",ax=axs[0])

sns.stripplot(x="label", y="x", data=avg_cluster, jitter=True,dodge=True, ax=axs[0], size=3)
axs[0].set_xticklabels(avg_cluster['label'].unique(), fontweight='bold',fontsize=14)
axs[0].set_xlabel('', fontweight='bold',fontsize=18,labelpad=18)
axs[0].set_ylabel('Distance x-axis (mm)', fontweight='bold',fontsize=18,labelpad=12)
# Hide the right and top spines
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].tick_params(axis='both', which='major', labelsize=14)
#lgnd = ax.legend(fontsize = 15, bbox_to_anchor= (1.1, 1.05), title="Axis", title_fontsize = 18, shadow = True, facecolor = 'white')
plt.suptitle('Source-Target distance after registration', y=1,fontproperties=fontTitle)
arr_img = plt.imread(r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/axial_frame.png', format='png')
newax = fig.add_axes([.82, 0.7, 0.15, 0.15], anchor='NE', zorder=1)
newax.imshow(arr_img)
newax.axis('off')
axs[0].annotate('', xy=(1.25,.4), xycoords='axes fraction', transform=axs[0].transAxes,xytext=(1.06,.4), arrowprops=dict(arrowstyle="<->", color='black',linewidth=2))
axs[0].text(1.12, .36,'X-axis', transform=axs[0].transAxes, fontsize=12, fontweight='bold')
axs[0].text(-.1, 1,'a)', transform=axs[0].transAxes, fontsize=18, fontweight='bold')


sns.violinplot(x="label", y="z",data=avg_cluster, inner="quartile", color=".6",ax=axs[1])
sns.stripplot(x="label", y="z", data=avg_cluster, jitter=True,dodge=True, ax=axs[1], size=3)
axs[1].set_xticklabels(avg_cluster['label'].unique(), fontweight='bold',fontsize=14)
axs[1].set_xlabel('Frame Fiducial Point', fontweight='bold',fontsize=18,labelpad=18)
axs[1].set_ylabel('Distance y-axis (mm)', fontweight='bold',fontsize=18,labelpad=12)
# Hide the right and top spines
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].tick_params(axis='both', which='major', labelsize=14)
arr_img = plt.imread(r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/axial_frame.png', format='png')
newax = fig.add_axes([.82, 0.25, 0.15, 0.15], anchor='NE', zorder=1)
newax.imshow(arr_img)
newax.axis('off')
axs[1].annotate('', xy=(1.05,.72), xycoords='axes fraction', xytext=(1.05,.32), arrowprops=dict(arrowstyle="<->", color='black',linewidth=2))
axs[1].text(1.02, .45,'Y-axis', transform=axs[1].transAxes, fontsize=12, fontweight='bold', rotation=90)
axs[1].text(-.1, 1,'b)', transform=axs[1].transAxes, fontsize=18, fontweight='bold')

sns.violinplot(x="label", y="y",data=avg_cluster, inner="quartile", color=".6",ax=axs[1])
sns.stripplot(x="label", y="y", data=avg_cluster, jitter=True,dodge=True, ax=axs[1], size=3)
axs[1].set_xticklabels(avg_cluster['label'].unique(), fontweight='bold',fontsize=14)
axs[1].set_xlabel('Frame Fiducial Point', fontweight='bold',fontsize=18,labelpad=18)
axs[1].set_ylabel('Distance y-axis (mm)', fontweight='bold',fontsize=18,labelpad=12)
# Hide the right and top spines
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].tick_params(axis='both', which='major', labelsize=14)
arr_img = plt.imread(r'/home/greydon/Documents/GitHub/phd_thesis/figures/static/axial_frame.png', format='png')
newax = fig.add_axes([.82, 0.25, 0.15, 0.15], anchor='NE', zorder=1)
newax.imshow(arr_img)
newax.axis('off')
axs[1].annotate('', xy=(1.05,.72), xycoords='axes fraction', xytext=(1.05,.32), arrowprops=dict(arrowstyle="<->", color='black',linewidth=2))
axs[1].text(1.02, .45,'Y-axis', transform=axs[1].transAxes, fontsize=12, fontweight='bold', rotation=90)
axs[1].text(-.1, 1,'b)', transform=axs[1].transAxes, fontsize=18, fontweight='bold')


plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.subplots_adjust(right=0.8)




#%%

plt.savefig(os.path.join(out_path,f"frame_error_distance_axes_violin.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"frame_error_distance_axes_violin.png"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"frame_error_distance_axes_violin_white.png"),transparent=False,dpi=400)
plt.close()


#%%
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.stats.multicomp as multi
import pingouin as pg

stat,p=sm.stats.anova_lm(error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[0]]['error'].values,error_data_melt[error_data_melt['system']==error_data_melt['system'].unique()[1]]['error'].values, type=1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
race_pairs = []

pg.intraclass_corr(data=error_data_melt, targets='subject',ratings='error',raters='system')

t=icc(np.c_[error_data_melt[error_data_melt['system']=='StealthStation']['error'], error_data_melt[error_data_melt['system']=='trajectoryGuide']['error']], 2, 'k')
aov = pg.anova(data=error_data_melt, dv='error', between='system', detailed=True)
print(aov)


posthoc = pg.pairwise_ttests(data=error_data_melt, dv='error', between='system',parametric=True, padjust='fdr_bh', effsize='hedges')
pg.print_table(posthoc, floatfmt='.3f')
mcDate = multi.MultiComparison(error_data_melt['error'], error_data_melt['system'])
Results = mcDate.tukeyhsd()
print(Results)

print(pairwise_tukeyhsd(error_data_melt['error'], error_data_melt['system']))

pg.intraclass_corr()
#%%
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
np.rmsValue(x0)
def rms(x):
    return np.sqrt(np.vdot(x, x)/x.size)


#%%


def IPN_icc(X, cse, typ):
	"""
	Computes the interclass correlations for indexing the reliability analysis 
	according to shrout & fleiss' schema.
	INPUT:
	x   - ratings data matrix, data whose columns represent different
		 ratings/raters & whose rows represent different cases or 
		 targets being measured. Each target is assumed too be a random
		 sample from a population of targets.
	cse - 1 2 or 3: 1 if each target is measured by a different set of 
		 raters from a population of raters, 2 if each target is measured
		 by the same raters, but that these raters are sampled from a 
		 population of raters, 3 if each target is measured by the same 
		 raters and these raters are the only raters of interest.
	typ - 'single' or 'k': denotes whether the ICC is based on a single
		 measurement or on an average of k measurements, where 
		 k = the number of ratings/raters.
	REFERENCE:
	Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
	reliability. Psychol Bull. 1979;86:420-428
	"""     

	[n, k] = np.shape(X)
	
	# mean per target
	mpt = np.mean(X, axis=1)
	# mean per rater/rating
	mpr = np.mean(X, axis=0)
	# get total mean
	tm = np.mean(X)
	# within target sum sqrs
	tmp = np.square(X - np.tile(mpt,(k,1)).T)
	WSS = np.sum(tmp)
	# within target mean sqrs
	WMS = float(WSS) / (n*(k - 1));
	# between rater sum sqrs
	RSS = np.sum(np.square(mpr - tm)) * n
	# between rater mean sqrs
	RMS = RSS / (float(k) - 1);
	# between target sum sqrs
	BSS = np.sum(np.square(mpt - tm)) * k
	# between targets mean squares
	BMS = float(BSS) / (n - 1)
	# residual sum of squares
	ESS = float(WSS) - RSS
	# residual mean sqrs
	EMS = ESS / ((k - 1) * (n - 1))

	if cse == 1:
		if typ == 'single':
			ICC = (BMS - WMS) / (BMS + (k - 1) * WMS)
		elif typ == 'k':
			ICC = (BMS - WMS) / BMS
		else:
			print("Wrong value for input type")

	elif cse == 2:
		if typ == 'single':
			ICC = (BMS - EMS) / (BMS + (k - 1) * EMS + k * (RMS - EMS) / n)
		elif typ == 'k':
			ICC = (BMS - EMS) / (BMS + (RMS - EMS) / n)
		else:
			print("Wrong value for input type")

	elif cse == 3:
		if typ == 'single':
			ICC = (BMS - EMS) / (BMS + (k - 1) * EMS)
		elif typ == 'k':
			ICC = (BMS - EMS) / BMS
		else:
			print("Wrong value for input type")

	else:
		print("Wrong value for input type")

	return ICC
























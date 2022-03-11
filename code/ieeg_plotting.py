#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

# set these paths to match what they are on your computer
input_fcsv_path=r'/media/veracrypt6/projects/iEEG/imaging/clinical/deriv/seega_coordinates'
input_transform_path=r'/media/veracrypt6/projects/iEEG/imaging/clinical/deriv/atlasreg'
output_plot_path=r'/home/greydon/Downloads'

# you can change the output filename but do not add a filetype extension (will be added later in the code)
output_plot_filename=r'subjects_space-MNI152NLin2009cSym_desc-affine_electrodes'

# do not modify these, based on the output naming of the pipeline
fcsv_filename='{isub}/{isub}_space-native_SEEGA.fcsv'
transform_filename='{isub}/{isub}_desc-affine_from-subject_to-MNI152NLin2009cSym_type-ras_xfm.txt'

# set the electrode name and contact numbers you want
electrodes={
	'sub-P053':{
		'LOFr':[1,2,3,4]
	},
	'sub-P055':{
		'LOFr':[1,2,3,4]
	},
	'sub-P056':{
		'LOFr':[1,2,3,4]
	},
	'sub-P057':{
		'LOFr':[1,2,3,4]
	},
	'sub-P059':{
		'LOFr':[1,2,3,4]
	},
	'sub-P060':{
		'LOFr':[1,2,3,4]
	},
	'sub-P061':{
		'LOFr':[1,2,3,4]
	}
}

# need to asign a color for each subject
cmap = plt.get_cmap('rainbow')
colors=cmap(np.linspace(0, 1, len(electrodes.keys())))

# this list will gather the transformed coordinates within the loop
tcoords = []

# this dictionary will gather information about each point (the specific color and desired label)
plot_settings={
	'color':[],
	'label':[]
}

# need to keep a running count of each subject that is looped through.
# This value will be used to assign thee color
isub_cnt=0

# for all the subjects (isub) in the dictionary 'electrodes'.
# Each subject defined in the dictionary is called a 'key', so 
# this loop is saying for all the keys in the dictionary perform
# the tasks inside the for loop
for isub in electrodes.keys():
	
	# load transform from subject to template.
	# The function 'format' here is used to replace the wildcard {isub} in the filename with
	# the specific subject in this iteration of the for loop
	sub2template= np.loadtxt(os.path.join(input_transform_path,transform_filename.format(isub=isub)))

	#read fcsv electrodes file, updating the filename for each subject
	df = pd.read_table(os.path.join(input_fcsv_path,fcsv_filename.format(isub=isub)),sep=',',header=2)
	
	# assign a color to the subject
	subject_color=colors[isub_cnt]
	
	# this is now calling the subject specific dirctionary within the parent dictionary 'electrodes'.
	# Now the keys are the desired electrodes you want plotted. When calling the subject specific dictionary
	# within the parent dictionary it would look like:
	# 
	# electrodes['sub-P060']={
	#     'LOFr':[1,2,3,4]
	# }
	#
	# So for all the electrodes in the patient dictionary, perform tasks within this for loop
	for ielectrode in electrodes[isub].keys():
		
		# For all the contacts in specific electrode list, run this for loop
		for icontact in electrodes[isub][ielectrode]:
			
			# build the contact label, used to search the fcsv file
			contact_label=ielectrode+str(icontact)

			# from the dataframe look for where 'label' is equal to the desired contact label
			# is the value is found, only return the x,y,z values
			coordinates=df[df['label']==contact_label][['x','y','z']].values
			
			# Check to make sure the coordinates were found, if the length of coordinates is 0
			# then the specific contact label was not found in the fcsv file
			if len(coordinates)>0:
				
				# Need to append 1 to the coordinates since the transform matrix is a 4x4 matrix
				# and current coordinates is only a 1x3 vector
				vector=np.append(coordinates[0],1)
				
				# take the inverse of the transform matrix (np.linalg.inv is a function that returns inverse) 
				# and use matrix multiplication (@) with the transpose of the coordinates vector 
				# (the .T here gets the transpose)
				tvec = np.linalg.inv(sub2template) @ vector.T
				
				# Append the new coordinates to the final transformed coordinates list, only taking the first
				# 3 values and ignoring the previous 1 that was appended
				tcoords.append(tvec[:3])
				
				# assign a color and label to the specific contact
				plot_settings['color'].append(subject_color)
				plot_settings['label'].append(isub.replace('sub-P0','') + '_' + contact_label)
	
	# Since this subject is now complete, update the subject count for the next
	# subject in the for loop
	isub_cnt+=1


# Call the'view_markers' plotting function from Nilearn with the specific inputs
#	tcoords: this is all the transformed coordinates with the shape (num_contacts,3), the columns being x,y,z in MNI
#	marker_size: size of the plotted marker, default is 6.0
#	marker_color: this is the color for each subject, which is a group of their contacts
#	marker_labels: the specific labels for each contact(the current label is subjectNumber_contactLabel; i.e. '60_LOFr1')
html_view = plotting.view_markers(
	tcoords,
	marker_size=6.0,
	marker_color=np.vstack(plot_settings['color']).tolist(),
	marker_labels=np.vstack(plot_settings['label']).tolist()
)

# This will open the plot in your browser window in real-time, to run this you need to remove the '#'
#html_view.open_in_browser()

# This will save the plot to the path specified above
html_view.save_as_html(os.path.join(output_plot_path,output_plot_filename+'.html'))


# The below function will save the 2D version of the glass brain as a PNG file
adjacency_matrix = np.zeros([len(tcoords),len(tcoords)])

display = plotting.plot_connectome(
	adjacency_matrix,
	tcoords,
	node_color=np.vstack(plot_settings['color']).tolist(),
	node_size=5
)

display.savefig(os.path.join(output_plot_path,output_plot_filename+'.png'),dpi=250)
display.close()














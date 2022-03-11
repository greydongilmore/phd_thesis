#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:20:52 2021

@author: greydon
"""

from bids import BIDSLayout
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import mer

bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'

layout = BIDSLayout(bids_dir)

isubject=layout.get_subjects()[-3]
iedf=edf_files[0]


for isubject in layout.get_subjects():
	edf_files=layout.get(subject=isubject, extension='.edf', return_type='filename')
	for iedf in edf_files:
		f = pyedflib.EdfReader(iedf)
		annots=f.readAnnotations()
		n = f.signals_in_file
		sigbufs = np.zeros((n, f.getNSamples()[0]))
		
		for i in np.arange(n):
			sigbufs[i, :] = f.readSignal(i)
		
		f.close()
		
		for idepth in range(len(numDepth)):
			index = patData[(patData['side']==numSides[iside]) & (patData['channel']==numChans[ichan]) & (patData['depth']==numDepth[idepth])].index[0]
			tempData = np.frombuffer(rawData[index])
			filen = outputChan + '/sub-' + str(subList[isub]) + '_side-' + numSides[iside] + '_depth-' + str(idepth+1) + '_channel-' + str(ichan+1)
			
			if idepth != (len(numDepth)-1):
				MAVtemp = (mer.MAVS(tempData, np.frombuffer(rawData[index+1])))
			else:
				MAVtemp = np.nan
			
			temp = [{'subject': subList[isub], 'side': numSides[iside], 'channel': numChans[ichan], 'depth': numDepth[idepth], 'labels': labels[index],'chanChosen': channelChosen,
					 'mav': mer.MAV(tempData), 
					 'mavSlope': MAVtemp,
					 'variance': mer.VAR(tempData), 
					 'mmav1': mer.MMAV1(tempData), 
					 'mmav2': mer.MMAV2(tempData), 
					 'rms': mer.RMS(tempData), 
					 'curveLength': mer.curveLen(tempData), 
					 'zeroCross': mer.zeroCross(tempData,10), 
					 'threshold': mer.threshold(tempData), 
					 'wamp': mer.WAMP(tempData,10), 
					 'ssi': mer.SSI(tempData), 
					 'power': mer.powerAVG(tempData), 
					 'peaks': mer.peaksNegPos(tempData), 
					 'tkeoTwo': mer.tkeoTwo(tempData), 
					 'tkeoFour': mer.tkeoFour(tempData),
					 'shapeF': mer.shapeFactor(tempData), 
					 'kurtosis': mer.KUR(tempData), 
					 'skew': mer.SKW(tempData), 
					 'crestF': mer.crestF(tempData), 
					 'meanF': mer.meanFrq(tempData,24000),
					 'frqRatio':mer.freqRatio(tempData, 24000),
					 'AvgPowerMU': mer.powerAVG(mer.butterBandpass(tempData, lowcut = 500, highcut = 1000, fs = 24000, order = 5)),
					 'AvgPowerSU': mer.powerAVG(mer.butterBandpass(tempData, lowcut = 1000, highcut = 3000, fs = 24000, order = 5)),
					 'entropy': mer.entropy(tempData),
					 'waveletStd': mer.wavlet(tempData, nLevels = 5, waveletName = 'db1', timewindow = False, windowSize = 0),
					 'rawData': rawData[index]}]
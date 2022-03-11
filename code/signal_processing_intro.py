#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:10:19 2021

@author: greydon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
from scipy import signal
from scipy.fftpack import fft

def get_fft_values(y_values, T, N, f_s):
	f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
	fft_values_ = fft(y_values)
	fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
	return f_values, fft_values


#%%

t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

x_value = np.linspace(0,t_n,N)
amplitudes = [4, 6, 8, 10, 14]
frequencies = [6.5, 5, 3, 1.5, 1]
y_values = [amplitudes[ii]*np.sin(2*np.pi*frequencies[ii]*x_value) for ii in range(0,len(amplitudes))]
composite_y_value = np.sum(y_values, axis=0)

f_values, fft_values = get_fft_values(composite_y_value, T, N, f_s)

colors = ['k', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("\nTime [s]", fontsize=16)
ax.set_ylabel("\nFrequency [Hz]", fontsize=16)
ax.set_zlabel("\nAmplitude", fontsize=16)

y_values_ = [composite_y_value] + list(reversed(y_values))
frequencies = [1, 1.5, 3, 5, 6.5]


for ii in range(0,len(y_values_)-1):
	signals = y_values_[ii]
	color = colors[ii]
	length = signals.shape[0]
	x=np.linspace(0,10,1000)
	y=np.array([frequencies[ii]]*length)
	z=signals
	if ii == 0:
		linewidth = 4
	else:
		linewidth = 2
	ax.plot(list(x), list(y), zs=list(z), linewidth=linewidth, color=color)
 
	x=[10]*75
	y=f_values[:75]
	z = fft_values[:75]*3
	ax.plot(list(x), list(y), zs=list(z), linewidth=2, color='red')
	
	plt.tight_layout()


#%% fft
fig = plt.figure(figsize=(8,8))

plt.plot(f_values, fft_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of the signal", fontsize=16)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.arange(0, 20, 0.1)
y = np.sin(x)
z = y*np.sin(x)

fig = plt.figure()
ax = plt.axes(projection='3d')

c = x + y

ax.scatter(x, y, z, c=c)

#%% power spectral density

data=np.loadtxt("/home/greydon/Downloads/thesis_resources/code/data.txt")

# Define sampling frequency and time vector
sf = 256
time = np.arange(data.size) / sf

# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.plot(time, data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.xlim([time.min(), time.max()])
plt.title('EEG data')
sns.despine()


# Define window length (4 seconds)
win = 4 * sf
freqs, psd = signal.welch(data, sf, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 8))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
sns.despine()

#%% extract specific band power

# Define delta lower and upper limits
low, high = 2, 8

# Find intersecting values in frequency vector
idx_delta = np.logical_and(freqs >= low, freqs <= high)

# Plot the power spectral density and fill the delta area
plt.figure(figsize=(8, 8))
plt.plot(freqs, psd, lw=2, color='k')
plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (uV^2 / Hz)')
plt.xlim([0, 10])
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram - Delta band")
sns.despine()


#%% average band power - integrate area under curve

from scipy.integrate import simps

# Frequency resolution
freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

# Compute the absolute power by approximating the area under the curve
delta_power = simps(psd[idx_delta], dx=freq_res)
print('Absolute delta power: %.3f uV^2' % delta_power)

#%% wavelet convolution

from mne.time_frequency import morlet

# Parameters
# Central frequency in Hz
cf = 13

# Number of oscillations
nc = 12

# Compute the wavelet
wlt = morlet(sf, [cf], n_cycles=nc)[0]

# Plot
t = np.arange(wlt.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(t, wlt)
plt.ylim(-0.4, 0.4)
plt.xlim(t[0], t[-1])
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude [a.u.]')

#%% convolution

# Convolve the wavelet and extract magnitude and phase
analytic = np.convolve(data, wlt, mode='same')
magnitude = np.abs(analytic)
phase = np.angle(analytic)

# Square and normalize the magnitude from 0 to 1 (using the min and max)
power = np.square(magnitude)
norm_power = (power - power.min()) / (power.max() - power.min())

# Define the threshold
thresh = 0.25

# Find supra-threshold values
supra_thresh = np.where(norm_power >= thresh)[0]

# Create vector for plotting purposes
val_spindles = np.nan * np.zeros(data.size)
val_spindles[supra_thresh] = data[supra_thresh]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(time, data, lw=1.5)
ax1.plot(time, val_spindles, color='indianred', alpha=.8)
ax1.set_xlim(0, time[-1])
ax1.set_ylabel('Voltage [uV]')
ax1.set_title('EEG signal')

ax2.plot(time, norm_power)
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Normalized wavelet power')
ax2.axhline(thresh, ls='--', color='indianred', label='Threshold')
ax2.fill_between(time, norm_power, thresh, where = norm_power >= thresh, color='indianred', alpha=.8)
plt.legend(loc='best')


#%%

Fs = 1 / dt               # Define the sampling frequency,
interval = int(Fs)        # ... the interval size,
overlap = int(Fs * 0.95)  # ... and the overlap intervals

                          # Compute the spectrogram
f, t, Sxx = spectrogram(data,fs=sf,nperseg=interval,noverlap=overlap)

pcolormesh(t, f, 10 * log10(Sxx),
               cmap='jet')# Plot the result
colorbar()                # ... with a color bar,
ylim([0, 70])             # ... set the frequency range,
xlabel('Time [s]')        # ... and label the axes
ylabel('Frequency [Hz]')
savefig('imgs/3-14')
show()





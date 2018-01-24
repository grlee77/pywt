#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pywt


x = np.linspace(0, 1, num=512)
data = np.sin(250 * np.pi * x**2)

wavelet = 'db2'
level = 4
order = "freq"  # other option is "natural"
interpolation = 'nearest'
cmap = plt.cm.cool

# Construct wavelet packet
wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
nodes = wp.get_level(level, order=order)
labels = [n.path for n in nodes]
values = np.array([n.data for n in nodes], 'd')
values = abs(values)

# Show signal and wavelet packet coefficients
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.4, bottom=.06, left=.09, right=.96, top=.92)
ax1.set_title("linchirp signal")
ax1.plot(x, data, 'b')
ax1.set_xlim(0, x[-1])

ax2.set_title("Wavelet packet coefficients at level %d" % level)
ax2.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
           origin="lower", extent=[0, 1, 0, len(values)])
ax2.set_yticks(np.arange(0.5, len(labels) + 0.5))
ax2.set_yticklabels(labels)

# Show spectrogram and wavelet packet coefficients
fig, (ax3, ax4) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.4)
ax3.specgram(data, NFFT=64, noverlap=32, Fs=2, cmap=cmap,
             interpolation='bilinear')
ax3.set_title("Spectrogram of signal")
ax4.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
           interpolation='nearest')
ax4.set_title("Wavelet packet coefficients")


plt.show()

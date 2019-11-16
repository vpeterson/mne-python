"""
===========================================================
Compute Sepctro-Spatial Decomposition (SDD) spatial filters
===========================================================

In this example, we will compute spatial filters for retaining
oscillatory brain activity and down-weighting 1/f background signals
as proposed by [1]_.
The idea is to learn spatial filters that separate oscillatory dynamics
from surrounding non-oscillatory noise based on the covariance in the
frequency band of interest and the noise covariance absed on surrounding
frequencies.

References
----------

.. [1] Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for
       reliable and fast extraction of neuronal EEG/MEG oscillations on the
       basis of spatio-spectral decomposition. NeuroImage, 55(4), 1528-1535.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np
import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne.utils import _time_mask
from mne.channels import read_layout



def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)

# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes

freqs_sig = 9, 12
freqs_noise = 8, 13

# Filter MEG data to focus on beta band
raw.pick_types(meg=True, ref_meg=True, eeg=False, eog=False)
raw_sig = raw.copy()
raw_sig.filter(*freqs_sig, l_trans_bandwidth=1, h_trans_bandwidth=1,
               fir_design='firwin')

raw_noise = raw.copy()
raw_noise.filter(*freqs_noise, l_trans_bandwidth=1, h_trans_bandwidth=1,
                 fir_design='firwin')

# remove signal from noise to further increase impact of surrounding freqs
raw_noise._data -= raw_sig._data

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1.5 second
picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
epochs_sig = Epochs(raw_sig, events, tmin=0., tmax=1.500, baseline=None,
                    picks=picks,
                    detrend=1, decim=4)

epochs_noise = Epochs(raw_noise, events, tmin=0., tmax=1.500, baseline=None,
                      picks=picks,
                      detrend=1, decim=4)

cov_sig = mne.compute_covariance(epochs_sig, method='oas')
cov_noise = mne.compute_covariance(epochs_noise, method='oas')


vals, vecs = eigh(cov_sig.data, cov_noise.data)


ssd_sources = vecs.T @ raw._data[picks]

psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)


def spectral_ratio_ssd(psd, freqs, freqs_sig, freqs_noise):
    """Spectral ratio measure
    See Nikulin 2011, Eq. (24)
    """
    sig_idx = freq_mask(freqs, *freqs_sig)
    noise_idx = freq_mask(freqs, *freqs_noise)
    ratio = psd[:, sig_idx].mean(axis=1) / psd[:, noise_idx].mean(axis=1)
    return ratio

pow_ratio = spectral_ratio_ssd(psd, freqs, freqs_sig, freqs_noise)

sorter = pow_ratio.argsort()

# plot spectral ratio (see Eq. 24 in Nikulin 2011)
plt.figure()
plt.plot(pow_ratio, color='black')
plt.plot(pow_ratio[sorter], color='orange', label='sorted eigenvalues')
plt.xlabel("Eigenvalue Index")
plt.ylabel(r"Power Ratio $\frac{P_f}{P_{sf}}$")
plt.legend()
plt.axhline(1, linestyle='--')

# Let's investigate spatila filter with max power ratio
# We willl first inspect the topographies
# According to Nikulin et al 2011 this is done
# by either inverting the filters (W^{-1}) or by multiplying the noise
# cov with the filters Eq. (22) (C_n W)^t.
# We'll explore both approaches to be sure all is fine.

max_idx = pow_ratio.argsort()[-1]
layout = read_layout('CTF151.lay')

A1 = np.linalg.pinv(vecs)
pattern1 = mne.EvokedArray(
    data=A1[max_idx, np.newaxis].T, info=epochs_sig.info)
pattern1.plot_topomap(times=[0.], units=dict(mag='A.U.'),
                      time_format='', layout=layout)

A2 = (cov_noise.data @ vecs).T
pattern2 = mne.EvokedArray(
    data=A2[max_idx, np.newaxis].T, info=epochs_sig.info)
pattern2.plot_topomap(times=[0.], units=dict(mag='A.U.'),
                      time_format='', layout=layout)

assert np.allclose(A1, A2)

# The topographies suggest that we picked up a parietal alpha generator

# Let's also look at tbe power spectrum of that source and compare it to
# to the power spectrum of the source with lowest SNR

min_idx = pow_ratio.argsort()[0]

below50 = freq_mask(freqs, 0, 50)
plt.figure()
plt.loglog(freqs[below50], psd[max_idx, below50], label='max SNR')
plt.loglog(freqs[below50], psd[min_idx, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()

# We can clearly see that the selected component enjoyes an SNR that is
# way above the average powe spectrum.
# The component on the bottom is mostly dominated by non-oscillatory signals.

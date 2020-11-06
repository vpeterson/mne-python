"""
===========================================================
Compute Sepctro-Spatial Decomposition (SDD) spatial filters
===========================================================
In this example, we will compute spatial filters for retaining
oscillatory brain activity and down-weighting 1/f background signals
as proposed by :footcite:`NikulinEtAl2011`.
The idea is to learn spatial filters that separate oscillatory dynamics
from surrounding non-oscillatory noise based on the covariance in the
frequency band of interest and the noise covariance absed on surrounding
frequencies.
References
----------
.. footbibliography::
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne.decoding import SSD
from mne.utils import _time_mask


def freq_mask(freqs, fmin, fmax):
    """Convenience function to select frequencies."""
    return _time_mask(freqs, fmin, fmax)


# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 110.).load_data()  # crop for memory purposes
raw.resample(sfreq=250)

picks_raw = mne.pick_types(
    raw.info, meg=True, eeg=False, ref_meg=False)
raw.pick(picks_raw)

freqs_sig = 9, 12
freqs_noise = 8, 13

# prepare data

ssd = SSD(info=raw.info,
          sort_by_spectral_ratio=False,  # True is recommended here.
          filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'),
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin'))
ssd.fit(X=raw.get_data())


# Let's investigate spatila filter with max power ratio.
# We will first inspect the topographies.
# According to Nikulin et al 2011 this is done.
# by either inverting the filters (W^{-1}) or by multiplying the noise
# cov with the filters Eq. (22) (C_n W)^t.
# We rely on the inversion approach here.

plt.close('all')
pattern = mne.EvokedArray(data=ssd.patterns_[:4].T,
                          info=ssd.info)
pattern.plot_topomap(units=dict(mag='A.U.'), time_format='')

# The topographies suggest that we picked up a parietal alpha generator.

# transform
ssd_sources = ssd.transform(X=raw.get_data())

# get psd of SSD-filtered signals
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)

# get spec_ratio information (already sorted)
spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

# plot spectral ratio (see Eq. 24 in Nikulin 2011)
plt.figure()
plt.plot(spec_ratio, color='black')
plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
plt.xlabel("Eigenvalue Index")
plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
plt.legend()
plt.axhline(1, linestyle='--')

# We can see that the inital sorting based on the eigenvalues
# was already quite good. However, when using few components only
# The sorting mighte make a difference.

# Let's also look at tbe power spectrum of that source and compare it to
# to the power spectrum of the source with lowest SNR.


below50 = freq_mask(freqs, 0, 50)
# for highlighting the freq. band of interest
bandfilt = freq_mask(freqs, freqs_sig[0], freqs_sig[1])
plt.figure()
plt.loglog(freqs[below50], psd[0, below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 10000, color='green', alpha=0.15)
plt.xlabel('log(frequency)')
plt.ylabel('log(power)')
plt.legend()

# We can clearly see that the selected component enjoys an SNR that is
# way above the average powe spectrum.

# Epoched data
# Although we suggest to use this method before epoching, there might be some
# situations in which data can only be treated by chunks

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=5.0, overlap=0.0)

# Epoch length is 1 second
epochs = Epochs(raw, events, tmin=0., tmax=5,
                baseline=None, preload=True)

ssd_epochs = SSD(info=epochs.info,
                 filt_params_signal=dict(l_freq=freqs_sig[0],
                                         h_freq=freqs_sig[1],
                                         l_trans_bandwidth=1,
                                         h_trans_bandwidth=1,
                                         fir_design='firwin'),
                 filt_params_noise=dict(l_freq=freqs_noise[0],
                                        h_freq=freqs_noise[1],
                                        l_trans_bandwidth=1,
                                        h_trans_bandwidth=1,
                                        fir_design='firwin'))
ssd_epochs.fit(X=epochs.get_data())

# epochs
pattern_epochs = mne.EvokedArray(data=ssd_epochs.patterns_[:4].T,
                                 info=ssd_epochs.info)
pattern_epochs.plot_topomap(units=dict(mag='A.U.'), time_format='')

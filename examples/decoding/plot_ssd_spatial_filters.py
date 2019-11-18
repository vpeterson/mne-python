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
from mne.decoding import TransformerMixin, BaseEstimator


def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)

# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes

freqs_sig = 9, 12
freqs_noise = 8, 13


class SSD(BaseEstimator, TransformerMixin):
    """Implement Spatio-Spectral Decomposision

    Parameters
    ----------
    filt_params_signal : dict
        Filtering for the frequencies of interst.
    filt_params_noise  : dict
        Filtering for the frequencies of non-interest.
    estimator : str
        Which covariance estimator to use. Defaults to "oas".
    n_components  : int, None
        How many components to keep. Default to 5.
    subtract_signal : bool
        Whether to subtract the noise from the signal to enhcane
        sptial filter.
    sort_by_spectral_ratio : bool
        Whether to sort by the spectral ratio. Defaults to True.

    """

    def __init__(self, filt_params_signal, filt_params_noise,
                 estimator='oas', n_components=None,
                 subtract_signal=True,
                 reject=None,
                 flat=None,
                 picks=None,
                 sort_by_spectral_ratio=True):
        """Initialize instance"""

        dicts = {"signal": filt_params_signal, "noise": filt_params_noise}
        for param, dd in [('l', 0), ('h', 0), ('l', 1), ('h', 1)]:
            key = ('signal', 'noise')[dd]
            if  param + '_freq' not in dicts[key]:
                raise ValueError(
                    "'%' must be defined in filter parameters for %s" % key)
            val = dicts[key][param + '_freq']
            if not isinstance(val, (int, float)):
                raise ValueError(
                    "Frequencies must be numbers, got %s" % type(val))

        self.freqs_signal = (filt_params_signal['l_freq'],
                             filt_params_signal['h_freq'])
        self.freqs_noise = (filt_params_noise['l_freq'],
                            filt_params_noise['h_freq'])
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.subtract_signal = subtract_signal
        self.picks = picks
        self.estimator = estimator
        self.n_components = n_components

    def fit(self, inst):
        """Fit"""
        if self.picks == None:
            self.picks_ = mne.pick_types(inst.info, meg=True, eeg=False, ref_meg=False)
        else:
            self.picks_ = self.picks
        if isinstance(inst, mne.io.base.BaseRaw):
            inst_signal = inst.copy()
            inst_signal.filter(picks=self.picks_, **self.filt_params_signal)

            inst_noise = inst.copy()
            inst_noise.filter(picks=self.picks_, **self.filt_params_noise)
            if self.subtract_signal:
                inst_noise._data[self.picks_] -= inst_signal._data[self.picks_]
            cov_signal = mne.compute_raw_covariance(
                inst_signal, picks=self.picks_, method=self.estimator)
            cov_noise = mne.compute_raw_covariance(inst_noise, picks=self.picks_,
                method=self.estimator)
            del inst_noise
            del inst_signal
        else:
            raise NotImplementedError()

        self.eigvals_, self.filters_ = eigh(cov_signal.data, cov_noise.data)
        self.patterns_ = np.linalg.pinv(self.filters_)

    def transform(self, data):
        if self.n_components is None:
            n_components = len(self.picks_)
        else:
            n_components = self.n_components
        return np.dot(self.filters_.T[:n_components], data[self.picks_])

    def apply(self, instance):
        pass

    def inverse_transform(self):
        raise NotImplementedError()


def spectral_ratio_ssd(psd, freqs, freqs_sig, freqs_noise):
    """Spectral ratio measure
    See Nikulin 2011, Eq. (24)
    """
    sig_idx = freq_mask(freqs, *freqs_sig)
    noise_idx = freq_mask(freqs, *freqs_noise)
    ratio = psd[:, sig_idx].mean(axis=1) / psd[:, noise_idx].mean(axis=1)
    return ratio


ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'),
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin'))

ssd.fit(raw.copy().crop(0, 120))


ssd_sources = ssd.transform(raw.get_data())

psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)

spec_ratio = spectral_ratio_ssd(psd, freqs, freqs_sig, freqs_noise)

sorter = spec_ratio.argsort()

# plot spectral ratio (see Eq. 24 in Nikulin 2011)
plt.figure()
plt.plot(spec_ratio, color='black')
plt.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
plt.xlabel("Eigenvalue Index")
plt.ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
plt.legend()
plt.axhline(1, linestyle='--')

# Let's investigate spatila filter with max power ratio.
# We willl first inspect the topographies.
# According to Nikulin et al 2011 this is done.
# by either inverting the filters (W^{-1}) or by multiplying the noise
# cov with the filters Eq. (22) (C_n W)^t.
# We rely on the inversion apprpach here.


max_idx = spec_ratio.argsort()[::-1][:4]
layout = read_layout('CTF151.lay')

pattern = mne.EvokedArray(
    data=np.linalg.pinv(ssd.filters_)[max_idx, :].T,
    info=mne.pick_info(raw.info, ssd.picks_))

pattern.plot_topomap(units=dict(mag='A.U.'),
                     time_format='', layout=layout)


# The topographies suggest that we picked up a parietal alpha generator.

# Let's also look at tbe power spectrum of that source and compare it to
# to the power spectrum of the source with lowest SNR.

min_idx = spec_ratio.argsort()[0]

below50 = freq_mask(freqs, 0, 50)
plt.figure()
plt.loglog(freqs[below50], psd[max_idx[0], below50], label='max SNR')
plt.loglog(freqs[below50], psd[min_idx, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()

# We can clearly see that the selected component enjoyes an SNR that is
# way above the average powe spectrum.

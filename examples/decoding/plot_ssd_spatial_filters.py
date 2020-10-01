"""
===========================================================
Compute Sepctro-Spatial Decomposition (SDD) spatial filters
===========================================================
In this example, we will compute spatial filters for retaining
oscillatory brain activity and down-weighting 1/f background signals
as proposed by :footcite:'NikulinEtAl2011'.
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
from scipy.linalg import eigh
import numpy as np
import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne.channels import read_layout
from mne.decoding import TransformerMixin, BaseEstimator

from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne.utils import _time_mask
from mne.cov import (_regularized_covariance, compute_raw_covariance)
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch


def freq_mask(freqs, fmin, fmax):
    """convenience function to select frequencies"""
    return _time_mask(freqs, fmin, fmax)


# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes
picks = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False)
sf = raw.info['sfreq']

freqs_sig = 9, 12
freqs_noise = 8, 13


class SSD(BaseEstimator, TransformerMixin):
    """
    SSD seeks at maximizing the power at a frequency band of interest while
    simultaneously minimizing it at the flanking (surrounding) frequency bins
    (considered noise). It extremizes the covariance matrices associated to
    signal and noise :footcite:'NikulinEtAl2011'.

    SSD can either be used as a dimensionality reduction method or a
    ‘denoised’ low rank factorization method :footcite:'HaufeEtAl2014'.

    Parameters
    ----------
    filt_params_signal : dict
        Filtering for the frequencies of interst.
    filt_params_noise  : dict
        Filtering for the frequencies of non-interest.
    sampling_freq : float
        sampling frequency (in Hz) of the recordings.
    estimator : float | str | None (default 'oas')
        Which covariance estimator to use.
        If not None (same as 'empirical'), allow regularization for
        covariance estimation. If float, shrinkage is used
        (0 <= shrinkage <= 1). For str options, estimator will be passed to
        method to :func:'mne.compute_covariance'.
    n_components : int | None (default None)
        The number of components to decompose the signals.
        If n_components is None no dimensionality reduction is made, and the
        transformed data is projected in the whole source space.
   picks: array| int | None  (default None)
       Indeces of good-channels. Can be the output of mne.pick_types.
   sort_by_spectral_ratio: bool (default True)
       if set to True, the components are sorted according
       to the spectral ratio.
       See , Eq. (24) in :footcite:'NikulinEtAl2011'
   return_filtered : bool (default False)
        If return_filtered is True, data is bandpassed and projected onto
        the SSD components.
   n_fft: int (default None)
       if sort_by_spectral_ratio is set to True, then the sources will be
       sorted accordinly to their spectral ratio which is calculated based on
       :func:'psd_array_welch' function. The n_fft parameter set the length of
       FFT used. See :func:'mne.time_frequency.psd_array_welch' for more
       information.
   cov_method_params : dict | None (default None)
        As in :func:'mne.decoding.SPoC'
        The default is None.
   rank : None | dict | ‘info’ | ‘full’
        As in :func:'mne.decoding.SPoC'
        This controls the rank computation that can be read from the
        measurement info or estimated from the data.
        See Notes of :func:'mne.compute_rank' for details.
        We recomend to use 'full' when working with epoched data.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, filt_params_signal, filt_params_noise, sampling_freq,
                 estimator='oas', n_components=None, picks=None,
                 sort_by_spectral_ratio=True, return_filtered=False,
                 n_fft=None, cov_method_params=None, rank=None):
        """Initialize instance"""

        dicts = {"signal": filt_params_signal, "noise": filt_params_noise}
        for param, dd in [('l', 0), ('h', 0), ('l', 1), ('h', 1)]:
            key = ('signal', 'noise')[dd]
            if param + '_freq' not in dicts[key]:
                raise ValueError(
                    "%s must be defined in filter parameters for %s"
                    % (param+'_freq', key))
            val = dicts[key][param + '_freq']
            if not isinstance(val, (int, float)):
                raise ValueError(
                    "Frequencies must be numbers, got %s" % type(val))
        # check freq bands
        if (filt_params_noise['l_freq'] > filt_params_signal['l_freq'] or
                filt_params_signal['h_freq'] > filt_params_noise['h_freq']):
            raise ValueError('Wrongly specified frequency bands!\n'
                    'The signal band-pass must be within the noise band-pass!')

        self.freqs_signal = (filt_params_signal['l_freq'],
                             filt_params_signal['h_freq'])
        self.freqs_noise = (filt_params_noise['l_freq'],
                            filt_params_noise['h_freq'])
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        if n_fft is None:
            self.n_fft = int(sampling_freq)
        else:
            self.n_fft = int(n_fft)
        self.picks_ = picks
        self.return_filtered = return_filtered
        self.estimator = estimator
        self.n_components = n_components
        self.rank = rank
        self.sampling_freq = sampling_freq
        self.cov_method_params = cov_method_params

    def _check_X(self, inst):
        """Check input data."""
        if not isinstance(inst, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(inst))
        if inst.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def filter_data(self, inst):
        if isinstance(inst, BaseRaw):
            inst_signal = inst.copy()
            inst_signal.filter(picks=self.picks_, **self.filt_params_signal)
            # noise
            inst_noise = inst.copy()
            inst_noise.filter(picks=self.picks_, **self.filt_params_noise)
            # subtract signal:
            inst_noise._data[self.picks_] -= inst_signal._data[self.picks_]
        else:
            if not isinstance(inst, np.ndarray):
                raise NotImplementedError()
            # data is epoched
            # part of the following code is copied from mne csp
            n_epochs, n_channels, n_samples = inst.shape
            # reshape for filtering
            X_aux = np.reshape(inst, [n_epochs, n_channels * n_samples])
            inst_signal = filter_data(
                X_aux, self.sampling_freq, **self.filt_params_signal)
            inst_noise = filter_data(
                X_aux, self.sampling_freq, **self.filt_params_noise)
            # subtract signal:
            inst_noise -= inst_signal
            # reshape to original shape
            inst_signal = np.reshape(
                inst_signal, [n_epochs, n_channels, n_samples])
            inst_noise = np.reshape(
                inst_noise, [n_epochs, n_channels, n_samples])
        return inst_signal, inst_noise

    def fit(self, inst):
        """Estimate the SSD decomposition on raw or epoched data.

        Parameters
        ----------
        inst : instance of Raw data, Epoch data or array with shape
            (n_epochs, n_channels, n_times)
            The data on which to estimate the SSD.

        Returns
        -------
        self : instance of SSD
            Returns the modified instance.
        """
        if isinstance(inst, BaseRaw):
            if self.picks_ is None:
                raise ValueError('picks should be provided')
            self.max_components = len(self.picks_)
            inst_signal, inst_noise = self.filter_data(inst)
            cov_signal = compute_raw_covariance(inst_signal, picks=self.picks_,
                method=self.estimator, rank=self.rank)
            cov_noise = compute_raw_covariance(inst_noise, picks=self.picks_,
                method=self.estimator, rank=self.rank)
            del inst_noise
            del inst_signal
        else:
            if isinstance(inst, BaseEpochs):
                X = inst.get_data()[:, self.picks]
            elif isinstance(inst, np.ndarray):
                X = inst
            else:
                raise NotImplementedError()
            self._check_X(X)
            # data is epoched
            # part of the following code is copied from mne csp
            n_epochs, n_channels, n_samples = X.shape
            self.max_components = n_channels
            X_s, X_n = self.filter_data(X)
            covs = np.empty((n_epochs, n_channels, n_channels))
            for ii, epoch in enumerate(X_s):
                covs[ii] = _regularized_covariance(epoch, reg=self.estimator,
                    method_params=self.cov_method_params, rank=self.rank)
            cov_signal = covs.mean(0)
            # Covariance matrix for the flanking frequencies (noise)
            # Estimate single trial covariance
            covs_n = np.empty((n_epochs, n_channels, n_channels))
            for ii, epoch in enumerate(X_n):
                covs_n[ii] = _regularized_covariance(epoch, reg=self.estimator,
                    method_params=self.cov_method_params,
                    rank=self.rank)
            cov_noise = covs_n.mean(0)
        eigvals_, eigvects_ = eigh(cov_signal.data, cov_noise.data)
        # sort in descencing order
        ix = np.argsort(eigvals_)[::-1]
        self.eigvals_ = eigvals_[ix]
        self.filters_ = eigvects_[:, ix]
        self.patterns_ = np.linalg.pinv(self.filters_)
        return self

    def spectral_ratio_ssd(self, ssd_sources):
        """
        Spectral ratio measure for best n_components selection
        See :footcite:'NikulinEtAl2011', Eq. (24).

        Parameters
        ----------
        ssd_sources : array
            data proyected on source space.

        Returns
        -------
        spec_ratio : array, shape (n_channels)
            array with the sprectal ratio value for each component
        sorter_spec : array, shape (n_channels)
            array of indeces for sorting spec_ratio.

        """
        psd, freqs = psd_array_welch(
            ssd_sources, sfreq=self.sampling_freq, n_fft=self.n_fft)
        sig_idx = _time_mask(freqs, *self.freqs_signal)
        noise_idx = _time_mask(freqs, *self.freqs_noise)
        if psd.ndim == 3:
            mean_sig = psd[:, :, sig_idx].mean(axis=2).mean(axis=0)
            mean_noise = psd[:, :, noise_idx].mean(axis=2).mean(axis=0)
            spec_ratio = mean_sig / mean_noise
        else:
            mean_sig = psd[:, sig_idx].mean(axis=1)
            mean_noise = psd[:, noise_idx].mean(axis=1)
            spec_ratio = mean_sig / mean_noise
        sorter_spec = spec_ratio.argsort()[::-1]
        return spec_ratio, sorter_spec

    def transform(self, inst):
        """Estimate epochs sources given the SSD filters.

        Parameters
        ----------
        inst : instance of Raw, Epochs or ndarray of shape
            (n_epochs, n_channels, n_times)
            The data to be processed. The instance is modified inplace.

        Returns
        -------
        X_ssd : instance of Raw, Epochs or np.array
            The processed data.
        """
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first call fit')
        if self.return_filtered:
            inst, _ = self.filter_data(inst)
        if isinstance(inst, BaseRaw):
            data = inst.get_data()
            X_ssd = np.dot(self.filters_.T, data[self.picks_])
        else:
            if isinstance(inst, BaseEpochs):
                X = inst.get_data()[:, self.picks]
            elif isinstance(inst, np.ndarray):
                X = inst
            else:
                raise NotImplementedError()
            self._check_X(X)
            # project data on source space
            X_ssd = np.asarray(
                    [np.dot(self.filters_.T, epoch) for epoch in X])
        if self.sort_by_spectral_ratio:
            self.spec_ratio, self.sorter_spec = self.spectral_ratio_ssd(
                ssd_sources=X_ssd)
            self.filters_ = self.filters_[:, self.sorter_spec]
            self.patterns_ = self.patterns_[self.sorter_spec]

            if isinstance(inst, BaseRaw):
                X_ssd = X_ssd[self.sorter_spec]
            else:
                X_ssd = X_ssd[:, self.sorter_spec, :]
            if self.n_components is None:
                n_components = self.max_components
                return X_ssd
            else:
                n_components = self.n_components
                if isinstance(inst, BaseRaw):
                    X_ssd = X_ssd[:n_components]
                else:
                    X_ssd = X_ssd[:, :n_components, :]
                return X_ssd

    def apply(self, inst):
        """Remove selected components from the signal.

        This procedure will reconstruct M/EEG signals from which the dynamics
        described by the excluded components is subtracted
        (denoised by low-rank factorization).
        See :footcite:'HaufeEtAl2014' for more information.

        The data is processed in place.

        Parameters
        ----------
        inst : instance of Raw, Epochs or ndarray
             The data to be processed. The instance is modified inplace.

        Returns
        -------
        X : instance of Raw, Epochs or ndarray
            The processed data.
        """
        X = np.empty_like(inst)
        X_ssd = self.transform(inst)
        pick_patterns = self.patterns_[:self.n_components].T
        if isinstance(inst, BaseRaw):
            X = np.dot(pick_patterns, X_ssd)
        else:
            X = np.asarray([np.dot(pick_patterns, epoch) for epoch in X_ssd])
        return X

    def inverse_transform(self):
        """
        Not implemented, see ssd.apply() instead.

        """
        raise NotImplementedError()


ssd = SSD(filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin'),
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                l_trans_bandwidth=1, h_trans_bandwidth=1,
                                fir_design='firwin'),
          sampling_freq=raw.info['sfreq'], picks=picks)
# fit
ssd.fit(raw.copy().crop(0, 120))
# transform
ssd_sources = ssd.transform(raw)
# get psd
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=sf, n_fft=4096)
# get spec_ratio information
spec_ratio = ssd.spec_ratio
sorter = ssd.sorter_spec

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
    data=ssd.patterns_[:4].T, info=mne.pick_info(raw.info, ssd.picks_))
pattern.plot_topomap(units=dict(mag='A.U.'),
                     time_format='', layout=layout)

# The topographies suggest that we picked up a parietal alpha generator.

# Let's also look at tbe power spectrum of that source and compare it to
# to the power spectrum of the source with lowest SNR.


below50 = freq_mask(freqs, 0, 50)
# for highlighting the freq. band of interest
bandfilt = freq_mask(freqs, freqs_sig[0], freqs_sig[1])
plt.figure()
plt.loglog(freqs[below50], psd[0, below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 100, color='green', alpha=0.5)
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()

# We can clearly see that the selected component enjoyes an SNR that is
# way above the average powe spectrum.

# Epoched data
# Although we suggest to use this method before epoching, there might be some
# situations in which data can only be treated by chunks

raw.pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1, baseline=None,
                    detrend=1, decim=1)
X = meg_epochs.get_data()
# fit
ssd.fit(X)
# transform
ssd_sources_epochs = ssd.transform(X)

# let's check enhanced picks:

psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources_epochs, sfreq=raw.info['sfreq'], n_fft=1000)
below50 = freq_mask(freqs, 0, 50)
bandfilt = freq_mask(freqs, freqs_sig[0], freqs_sig[1])
# take the mean along epochs
psd = psd.mean(axis=0)
plt.figure()
plt.loglog(freqs[below50], psd[0, below50], label='max SNR')
plt.loglog(freqs[below50], psd[-1, below50], label='min SNR')
plt.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
plt.fill_between(freqs[bandfilt], 0, 100, color='green', alpha=0.5)
plt.xlabel("log(frequency)")
plt.ylabel("log(power)")
plt.legend()


# the picks can also be enhanced with epoched data!

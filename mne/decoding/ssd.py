# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD (3-clause)

import numpy as np
from scipy.linalg import eigh
from ..io.pick import channel_type
from ..filter import filter_data
from ..cov import _regularized_covariance
from . import TransformerMixin, BaseEstimator
from ..time_frequency import psd_array_welch
from ..utils import _time_mask


class SSD(BaseEstimator, TransformerMixin):
    """
    SSD seeks at maximizing the power at a frequency band of interest while
    simultaneously minimizing it at the flanking (surrounding) frequency bins
    (considered noise). It extremizes the covariance matrices associated to
    signal and noise :footcite:`NikulinEtAl2011`.

    SSD can either be used as a dimensionality reduction method or a
    ‘denoised’ low rank factorization method :footcite:`HaufeEtAl2014`.

    Parameters
    ----------
    info : instance of mne.Info
        The info object containing the channel and sampling information.
        It must match the input data.
    filt_params_signal : dict
        Filtering for the frequencies of interest.
    filt_params_noise  : dict
        Filtering for the frequencies of non-interest.
    estimator : float | str | None (default 'oas')
        Which covariance estimator to use.
        If not None (same as 'empirical'), allow regularization for
        covariance estimation. If float, shrinkage is used
        (0 <= shrinkage <= 1). For str options, estimator will be passed to
        method to :func:`mne.compute_covariance`.
    n_components : int | None (default None)
        The number of components to extract from the signal.
        If n_components is None, no dimensionality reduction is applied, and the
        transformed data is projected in the whole source space.
    sort_by_spectral_ratio: bool (default False)
       if set to True, the components are sorted according
       to the spectral ratio.
       See Eq. (24) in :footcite:`NikulinEtAl2011`
    return_filtered : bool (default True)
        If return_filtered is True, data is bandpassed and projected onto
        the SSD components.
    n_fft: int (default None)
       if sort_by_spectral_ratio is set to True, then the sources will be
       sorted accordinly to their spectral ratio which is calculated based on
       :func:`psd_array_welch` function. The n_fft parameter set the length of
       FFT used. See :func:`mne.time_frequency.psd_array_welch` for more
       information.
    cov_method_params : dict | None (default None)
        As in :func:`mne.decoding.SPoC`
        The default is None.
    rank : None | dict | ‘info’ | ‘full’
        As in :func:`mne.decoding.SPoC`
        This controls the rank computation that can be read from the
        measurement info or estimated from the data.
        See Notes of :func:`mne.compute_rank` for details.
        We recomend to use 'full' when working with epoched data.

    Attributes
    ----------
    filters_ : array, shape(n_channels, n_components)
        The spatial filters to be multipled with the signal.
    patterns_ : array, shape(n_components, n_channels)
        The patterns for reconstructing the signal from the filtered data.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, info, filt_params_signal, filt_params_noise,
                 estimator='oas', n_components=None,
                 sort_by_spectral_ratio=True, return_filtered=False,
                 n_fft=None, cov_method_params=None, rank=None):
        """Initialize instance"""

        dicts = {"signal": filt_params_signal, "noise": filt_params_noise}
        for param, dd in [('l', 0), ('h', 0), ('l', 1), ('h', 1)]:
            key = ('signal', 'noise')[dd]
            if param + '_freq' not in dicts[key]:
                raise ValueError(
                    "%s must be defined in filter parameters for %s"
                    % (param + '_freq', key))
            val = dicts[key][param + '_freq']
            if not isinstance(val, (int, float)):
                raise ValueError(
                    "Frequencies must be numbers, got %s" % type(val))
        # check freq bands
        if (filt_params_noise['l_freq'] > filt_params_signal['l_freq'] or
                filt_params_signal['h_freq'] > filt_params_noise['h_freq']):
            raise ValueError('Wrongly specified frequency bands!\n'
                    'The signal band-pass must be within the noise band-pass!')
        ch_types = {channel_type(info, ii)
                    for ii in range(info['nchan'])}
        if len(ch_types) > 1:
            raise ValueError("At this point SSD only supports fitting "
                             "single channel types. Your info has %i types" %
                             (len(ch_types)))
        self.info = info
        self.freqs_signal = (filt_params_signal['l_freq'],
                             filt_params_signal['h_freq'])
        self.freqs_noise = (filt_params_noise['l_freq'],
                            filt_params_noise['h_freq'])
        self.filt_params_signal = filt_params_signal
        self.filt_params_noise = filt_params_noise
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        if n_fft is None:
            self.n_fft = int(self.info['sfreq'])
        else:
            self.n_fft = int(n_fft)
        self.return_filtered = return_filtered
        self.estimator = estimator
        self.n_components = n_components
        self.rank = rank
        self.cov_method_params = cov_method_params

    def _check_X(self, X):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        if X.ndim < 2:
            raise ValueError('X must have at least 2 dimensions.')
        elif X.ndim > 3:
            raise ValueError('X must have at most 3 dimensions.')

        n_chan = 0
        if X.ndim == 2:
            n_chan = X.shape[0]
        elif X.ndim == 3:
            n_chan = X.shape[1]
        if n_chan != self.info['nchan']:
            raise ValueError('Info must match the input data.'
                             'Found %i channels but expected %i.' % (
                                n_chan, self.info['nchan']))

    def fit(self, X, y=None):
        """Estimate the SSD decomposition on raw or epoched data.

        Parameters
        ----------
        X : array, shape (n_channels, n_times) | shape (n_epochs,
                n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continous data or 3D array obtained from epoched
            data.
        y : None | array, shape (n_samples,)
                    Used for scikit-learn compatibility.
        Returns
        -------
        self : instance of SSD
            Returns the modified instance.
        """
        self._check_X(X)
        if X.ndim == 2:
            X_aux = X[self.picks_, :].copy()

        elif X.ndim == 3:
            X_aux = X[:, self.picks_, :].copy()
        else:
            raise NotImplementedError()

        X_signal = filter_data(
            X_aux, self.info['sfreq'], **self.filt_params_signal)
        X_noise = filter_data(
            X_aux, self.info['sfreq'], **self.filt_params_noise)

        X_noise -= X_signal 
        if X.ndim == 3:
            X_signal = np.hstack(X_signal)
            X_noise = np.hstack(X_noise)

        cov_signal = _regularized_covariance(
            X_signal, reg=self.estimator, method_params=self.cov_method_params,
            rank=self.rank, info=self.info)
        cov_noise = _regularized_covariance(
            X_noise, reg=self.estimator, method_params=self.cov_method_params,
            rank=self.rank, info=self.info)
   
        eigvals_, eigvects_ = eigh(cov_signal, cov_noise)
        # sort in descencing order
        ix = np.argsort(eigvals_)[::-1]
        self.eigvals_ = eigvals_[ix]
        self.filters_ = eigvects_[:, ix]
        self.patterns_ = np.linalg.pinv(self.filters_)
        return self

    def transform(self, X, y=None):
        """Estimate epochs sources given the SSD filters.

        Parameters
        ----------
        X : array, shape (n_channels, n_times) | shape (n_epochs,
                n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continous data or 3D array obtained from epoched
            data.
        y : None | array, shape (n_samples,)
                    Used for scikit-learn compatibility.

        Returns
        -------
        X_ssd : instance of Raw, Epochs or np.array
            The processed data.
        """
        self._check_X(X)
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first call fit')
        if X.ndim == 2:
            X_ssd = np.dot(self.filters_.T, X[self.picks_])
        elif X.ndim == 3:
            # project data on source space
            X_ssd = np.empty_like(X)
            for ii, x in enumerate(X[:, self.picks_]):
                X_ssd[ii] = np.dot(self.filters_.T, x)
        else:
            raise NotImplementedError()

        # We assume that ordering by spectral ratio is more important
        # than the inital ordering. This is why we apply component picks
        # after ordering.
        sorter_spec = Ellipsis
        if self.sort_by_spectral_ratio:
            _, sorter_spec = self.get_spectral_ratio(
                ssd_sources=X_ssd)
        if X.ndim == 2:
            X_ssd = X_ssd[sorter_spec][:self.n_components]
        else:
            X_ssd = X_ssd[:, sorter_spec, :][:, :self.n_components, :]
        return X_ssd

    def get_spectral_ratio(self, ssd_sources):
        """Get the spectal signal-to-noise ratio for each spatial filter.

        Spectral ratio measure for best n_components selection
        See :footcite:`NikulinEtAl2011`, Eq. (24).

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
            ssd_sources, sfreq=self.info['sfreq'], n_fft=self.n_fft)
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

    def apply(self):
        """
        Not implemented, see ssd.apply() instead.

        """
        raise NotImplementedError()
       

    def inverse_transform(self, X):
        """Remove selected components from the signal.

        This procedure will reconstruct M/EEG signals from which the dynamics
        described by the excluded components is subtracted
        (denoised by low-rank factorization).
        See :footcite:`HaufeEtAl2014` for more information.

        The data is processed in place.

        Parameters
        ----------
        X : array, shape (n_channels, n_times) | shape (n_epochs,
                n_channels, n_times)
            The input data from which to estimate the SSD. Either 2D array
            obtained from continous data or 3D array obtained from epoched
            data.

        Returns
        -------
        X : array
            The processed data.
        """
        X_ssd = self.transform(X)
        sorter_spec = Ellipsis
        if self.sort_by_spectral_ratio:
            _, sorter_spec = self.get_spectral_ratio(ssd_sources=X_ssd)

        pick_patterns = self.patterns_[sorter_spec, :self.n_components].T
        if X.ndim == 2:
            X = np.dot(pick_patterns, X_ssd)
        else:
            X = np.asarray([np.dot(pick_patterns, epoch) for epoch in X_ssd])
        return X
        
    
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from mne.datasets.fieldtrip_cmc import data_path
from mne import io, Epochs
from mne.time_frequency import psd_array_welch
from mne.decoding.ssd import SSD
from mne.utils import requires_sklearn
from mne.filter import filter_data

fname = data_path() + '/SubjectCMC.ds'

freqs_sig = 9, 12
freqs_noise = 8, 13

def simulate_data(freqs_sig = [9, 12], n_trials=100, n_channels=20,
                  n_samples=500, samples_per_second=250,
                  n_components=5, SNR=0.05, random_state=42):
    """Simulate data according to an instantaneous mixin model.

    Data are simulated in the statistical source space, where n=n_components
    sources contain the peak of interest,
    """
    rs = np.random.RandomState(random_state)

    filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')

    # generate a orthogonal mixin matrix
    mixing_mat = np.linalg.svd(rs.randn(n_channels, n_channels))[0]
    # define sources
    S = rs.randn(n_trials*n_samples, n_channels)
    # filter source in the specific freq. band of interest
    # S = filter_data(S.T, samples_per_second, **filt_params_signal ).T
    # mix data
    X_s = (mixing_mat[:,:n_components] @ filter_data(S[:,:n_components].T, samples_per_second, **filt_params_signal )).T   
    X_n = (mixing_mat[:,n_components:] @ S[:,n_components:].T).T 
    # add noise
    X_s = X_s / np.linalg.norm(X_s, 'fro')
    X_n = X_n / np.linalg.norm(X_n, 'fro')
    X = SNR * X_s + (1-SNR) * X_n
    return X, mixing_mat, S

@pytest.mark.slowtest
def test_csp():
    """Test Common Spatial Patterns algorithm on raw data."""
    raw = io.read_raw_ctf(fname)
    raw.crop(50., 110.).load_data()  # crop for memory purposes
    raw.resample(sfreq=250)
    info = raw.info
    data = raw.get_data()
    n_channels = data.shape[1]

    # Init
    # components no int
    pytest.raises(ValueError, SSD, n_components='foo')
    # freq no int
    for freq in ['foo', 1, 2]:
        filt_params_signal = dict(l_freq=freq, h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')
        filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin')
        ssd = SSD(info, filt_params_signal, filt_params_noise)
        pytest.raises(ValueError, ssd.fit, raw.get_data())
    # filt param no dict
    filt_params_signal = freqs_sig
    filt_params_noise = freqs_noise
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    pytest.raises(ValueError, ssd.fit, raw.get_data())

    # data type
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin')
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    pytest.raises(ValueError, ssd.fit, raw)    
    
    # Fit
    n_components=10
    filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')
    filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin')
    ssd = SSD(info, filt_params_signal, filt_params_noise,
              n_components=n_components)
    ssd.fit(data)

    assert (ssd.filters_.shape == (n_channels, n_channels))
    assert (ssd.patterns_.shape == (n_channels, n_channels))

    # Transform
    X_ssd = ssd.fit_transform(data)
    assert (X_ssd.shape[1] == n_components)
    # I'd like test here power ratio ordering

    # Apply
    X_denoised = ssd.apply(data)
    assert_array_almost_equal(X_denoised, data)

    # Check mixing matrix on simulated data
    X, A, S = simulate_data()

    # fit ssd
    n_components = 5 # we now that are 5
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin')
    ssd = SSD(None, filt_params_signal, filt_params_noise,
              n_components=n_components)
    ssd.fit(X)

    # check the first pattern match the mixing matrix
    # the sign might change
    corr = np.abs(np.corrcoef(ssd.patterns_[0, :].T, A[:, 0])[0, 1])
    assert np.abs(corr) > 0.99
    # check psd
    out = ssd.transform(X)
    psd_out, _ = psd_array_welch(out, sfreq=250, n_fft=250)
    psd_S, _ = psd_array_welch(S, sfreq=250, n_fft=250)
    corr = np.abs(np.corrcoef(psd_out[:, 0], psd_S[:, 0])[0, 1])
    assert np.abs(corr) > 0.95

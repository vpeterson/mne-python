# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from mne.datasets.fieldtrip_cmc import data_path
from mne import io, Epochs, read_events, pick_types
from mne.decoding.ssd import SSD
from mne.utils import requires_sklearn
from mne.filter import filter_data

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
# if stop is too small pca may fail in some cases, but we're okay on this file
start, stop = 0, 8
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
    return X, mixing_mat

@pytest.mark.slowtest
def test_csp():
    """Test Common Spatial Patterns algorithm on epochs."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[2:12:3]  # subselect channels -> disable proj!
    raw.add_proj([], remove_existing=True)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, proj=False)
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]
    y = epochs.events[:, -1]

    # Init
    pytest.raises(ValueError, SSD, n_components='foo')
    for freq in ['foo', 1, 2]:
        filt_params_signal=dict(l_freq=freq, h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1,
                                  fir_design='firwin')
        filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1,
                                 fir_design='firwin')
        ssd = SSD(info, filt_params_signal, filt_params_noise )
        pytest.raises(ValueError, ssd.fit, epochs_data, epochs.events[:, -1])
    for reg in ['oas', 'ledoit_wolf', 0, 0.5, 1.]:
        CSP(reg=reg, norm_trace=False)
    for cov_est in ['foo', None]:
        pytest.raises(ValueError, CSP, cov_est=cov_est, norm_trace=False)
    with pytest.raises(TypeError, match='instance of bool'):
        CSP(norm_trace='foo')
    for cov_est in ['concat', 'epoch']:
        CSP(cov_est=cov_est, norm_trace=False)

    n_components = 3
    # Fit
    for norm_trace in [True, False]:
        csp = CSP(n_components=n_components, norm_trace=norm_trace)
        csp.fit(epochs_data, epochs.events[:, -1])

    assert_equal(len(csp.mean_), n_components)
    assert_equal(len(csp.std_), n_components)

    # Transform
    X = csp.fit_transform(epochs_data, y)
    sources = csp.transform(epochs_data)
    assert (sources.shape[1] == n_components)
    assert (csp.filters_.shape == (n_channels, n_channels))
    assert (csp.patterns_.shape == (n_channels, n_channels))
    assert_array_almost_equal(sources, X)

    # Test data exception
    pytest.raises(ValueError, csp.fit, epochs_data,
                  np.zeros_like(epochs.events))
    pytest.raises(ValueError, csp.fit, epochs, y)
    pytest.raises(ValueError, csp.transform, epochs)

    # Test plots
    epochs.pick_types(meg='mag')
    cmap = ('RdBu', True)
    components = np.arange(n_components)
    for plot in (csp.plot_patterns, csp.plot_filters):
        plot(epochs.info, components=components, res=12, show=False, cmap=cmap)

    # Test with more than 2 classes
    epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, picks=picks,
                    event_id=dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4),
                    baseline=(None, 0), proj=False, preload=True)
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]

    n_channels = epochs_data.shape[1]
    for cov_est in ['concat', 'epoch']:
        csp = CSP(n_components=n_components, cov_est=cov_est, norm_trace=False)
        csp.fit(epochs_data, epochs.events[:, 2]).transform(epochs_data)
        assert_equal(len(csp._classes), 4)
        assert_array_equal(csp.filters_.shape, [n_channels, n_channels])
        assert_array_equal(csp.patterns_.shape, [n_channels, n_channels])

    # Test average power transform
    n_components = 2
    assert (csp.transform_into == 'average_power')
    feature_shape = [len(epochs_data), n_components]
    X_trans = dict()
    for log in (None, True, False):
        csp = CSP(n_components=n_components, log=log, norm_trace=False)
        assert (csp.log is log)
        Xt = csp.fit_transform(epochs_data, epochs.events[:, 2])
        assert_array_equal(Xt.shape, feature_shape)
        X_trans[str(log)] = Xt
    # log=None => log=True
    assert_array_almost_equal(X_trans['None'], X_trans['True'])
    # Different normalization return different transform
    assert (np.sum((X_trans['True'] - X_trans['False']) ** 2) > 1.)
    # Check wrong inputs
    pytest.raises(ValueError, CSP, transform_into='average_power', log='foo')

    # Test csp space transform
    csp = CSP(transform_into='csp_space', norm_trace=False)
    assert (csp.transform_into == 'csp_space')
    for log in ('foo', True, False):
        pytest.raises(ValueError, CSP, transform_into='csp_space', log=log,
                      norm_trace=False)
    n_components = 2
    csp = CSP(n_components=n_components, transform_into='csp_space',
              norm_trace=False)
    Xt = csp.fit(epochs_data, epochs.events[:, 2]).transform(epochs_data)
    feature_shape = [len(epochs_data), n_components, epochs_data.shape[2]]
    assert_array_equal(Xt.shape, feature_shape)

    # Check mixing matrix on simulated data
    y = np.array([100] * 50 + [1] * 50)
    X, A = simulate_data(y)

    for cov_est in ['concat', 'epoch']:
        # fit csp
        csp = CSP(n_components=1, cov_est=cov_est, norm_trace=False)
        csp.fit(X, y)

        # check the first pattern match the mixing matrix
        # the sign might change
        corr = np.abs(np.corrcoef(csp.patterns_[0, :].T, A[:, 0])[0, 1])
        assert np.abs(corr) > 0.99

        # check output
        out = csp.transform(X)
        corr = np.abs(np.corrcoef(out[:, 0], y)[0, 1])
        assert np.abs(corr) > 0.95



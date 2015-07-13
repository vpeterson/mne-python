# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Yousra Bekhti <>
#
# License: BSD (3-clause)
from __future__ import division
import copy

import numpy as np

from ..io.pick import pick_channels_cov
from ..forward import apply_forward
from ..utils import check_random_state, verbose, _time_mask

#TODO: Add back in linear filter for both epoch and noise generation (like
#      in simulate evoked)

def generate_epochs(fwd, stcs, info, cov, snr=3, tmin=None, tmax=None,
                    random_state=None):
    """Generate noisy evoked data

    Parameters
    ----------
    fwd : dict
        A forward solution
    stc : SourceEstimate | list of SourceEstimate
        The source time courses (as generated by simulation.simulate_stc, for
        example)
    info : dict
        Info dictionary
    cov : Covariance object
        The noise covariance
    snr : float | list, len(stc)
        signal to noise ratio in dB. It corresponds to
        10 * log10(var(signal) / var(noise)).
    tmin : float | None
        start of time interval to estimate SNR. If None first time point is
        used.
    tmax : float
        start of time interval to estimate SNR. If None last time point
        is used.
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    epochs : Epochs object
        The simulated data
    """
    data_arrays = mne.EpochsArray([_apply_forward(fwd, stc, tmin, tmax)
                                   for stc in stcs])
    # TODO: Get events object
    events = []

    # TODO: check if other default params should be included
    epochs = mne.EpochsArray(data_arrays, info, events, tmin)

    # Generate and add in noise for evoked objects
    noise = generate_noise_evoked(evoked, cov, random_state)
    evoked_noise = add_noise_evoked(evoked, noise, snr, tmin=tmin, tmax=tmax)
    return evoked_noise


def generate_noise_evoked(epochs, cov, iir_filter=None, random_state=None):
    """Creates noise as a multivariate Gaussian

    The spatial covariance of the noise is given from the cov matrix.

    Parameters
    ----------
    epochs : mne.epochs
        Epochs object to be used as template
    cov : Covariance object
        The noise covariance
    random_state : None | int | np.random.RandomState
        Random generator state

    Returns
    -------
    noise : evoked object
        An mne.epochs object
    """
    epo_noise = copy.deepcopy(epochs)
    n_channels = np.zeros(epo_noise.info['nchan'])
    n_samples = len(evoked.times)

    noise_cov = pick_channels_cov(cov, include=epo_noise.info['ch_names'])
    c = np.diag(noise_cov.data) if noise_cov['diag'] else noise_cov.data

    # Create epo_noise
    rng = check_random_state(random_state)
    epo_noise.data = rng.multivariate_normal(n_channels, c, n_samples).T

    return noise


def add_noise_epochs(epochs, noise, snr, tmin=None, tmax=None):
    """Adds noise to epochs object with specified SNR(s).

    SNR(s) computed in the interval from tmin to tmax.

    Parameters
    ----------
    epochs : Epochs object
        An instance of epochs signal
    noise : Epochs object
        An instance of epochs object filled purely with noise
    snr : float | len()
        signal to noise ratio in dB. It corresponds to
        10 * log10( var(signal) / var(noise))
    tmin : float
        start time before event
    tmax : float
        end time after event

    Returns
    -------
    epochs_noised : Epochs  object
        An instance of epochs corrupted by noise
    """
    epochs = copy.deepcopy(epochs)

    for epo, noise_epo in zip(epochs, noise_epo):
        tmask = _time_mask(evoked.times, tmin, tmax)
        tmp = 10 * np.log10(np.mean((epo.data[:, tmask] ** 2).ravel()) /
                            np.mean((noise_epo.data ** 2).ravel()))
        noise_epo.data = 10 ** ((tmp - snr) / 20) * noise_epo.data
        epo.data += noise_epo.data

    return evoked

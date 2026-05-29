"""
qkd/search.py
-------------
Distance and noise sweep functions.

These functions answer the central question of the simulation:
given a noise level and channel type, what is the maximum distance
at which the system can still generate a provably secure key?

The search uses a linear scan from 0 km to MAX_DIST_SEARCH in steps
of DIST_STEP. QBER is monotonically increasing with distance for the
channel models used here, so the scan always terminates at the correct
crossing point without risk of missing a later failure.
"""

import numpy as np
from atmos_qkd.constants import (
    DIST_STEP, MAX_DIST_SEARCH, SECURITY_LIMIT,
    SAMPLES, NOISE_MIN, NOISE_MAX
)
from atmos_qkd.qkd.qber import get_qber_fibre, get_qber_atmospheric


def find_max_distance_fibre(noise_std, protected=False):
    """
    Maximum secure distance for the fibre channel at a given noise level.

    Scans from 0 to MAX_DIST_SEARCH km and returns the first distance
    where QBER exceeds the Shor-Preskill security threshold. If QBER
    stays below the threshold across the full search range, returns
    MAX_DIST_SEARCH as the result.

    Parameters
    ----------
    noise_std  : float   polarization rotation std dev in radians
    protected  : bool    twirling protection active

    Returns
    -------
    max_dist : float   maximum secure distance in km
    """
    for d in np.arange(0, MAX_DIST_SEARCH, DIST_STEP):
        if get_qber_fibre(d, noise_std, protected) > SECURITY_LIMIT:
            return float(d)
    return float(MAX_DIST_SEARCH)


def find_max_distance_atmospheric(noise_std, Cn2,
                                  protected=False, **kwargs):
    """
    Maximum secure distance for an atmospheric free-space channel.

    Same scan logic as find_max_distance_fibre() with the atmospheric
    QBER function substituted in. The Cn2 parameter selects the
    turbulence regime.

    Parameters
    ----------
    noise_std  : float   polarization rotation std dev in radians
    Cn2        : float   turbulence strength in m^-2/3
    protected  : bool    twirling protection active
    **kwargs             forwarded to get_qber_atmospheric

    Returns
    -------
    max_dist : float   maximum secure distance in km
    """
    for d in np.arange(0, MAX_DIST_SEARCH, DIST_STEP):
        if get_qber_atmospheric(d, noise_std, Cn2,
                                protected, **kwargs) > SECURITY_LIMIT:
            return float(d)
    return float(MAX_DIST_SEARCH)


def sweep_noise_fibre(n_samples=SAMPLES, protected=False):
    """
    Full noise sweep for the fibre channel.

    Returns the noise levels and corresponding maximum secure
    distances — the data that produces the red and blue lines
    in the paper's Figure 4.

    Parameters
    ----------
    n_samples  : int    number of noise levels to evaluate
    protected  : bool   twirling protection active

    Returns
    -------
    noise_levels  : np.ndarray   noise std dev values in radians
    max_distances : np.ndarray   maximum secure distance at each level in km
    """
    noise_levels  = np.linspace(NOISE_MIN, NOISE_MAX, n_samples)
    max_distances = np.array([
        find_max_distance_fibre(n, protected) for n in noise_levels
    ])
    return noise_levels, max_distances


def sweep_noise_atmospheric(Cn2, n_samples=SAMPLES,
                             protected=False, **kwargs):
    """
    Full noise sweep for an atmospheric channel.

    Parameters
    ----------
    Cn2        : float   turbulence strength in m^-2/3
    n_samples  : int     number of noise levels to evaluate
    protected  : bool    twirling protection active
    **kwargs             forwarded to find_max_distance_atmospheric

    Returns
    -------
    noise_levels  : np.ndarray   noise std dev values in radians
    max_distances : np.ndarray   maximum secure distance at each level in km
    """
    noise_levels  = np.linspace(NOISE_MIN, NOISE_MAX, n_samples)
    max_distances = np.array([
        find_max_distance_atmospheric(n, Cn2, protected, **kwargs)
        for n in noise_levels
    ])
    return noise_levels, max_distances


def sweep_distance_qber(noise_std, Cn2=None, protected=False,
                        n_points=500):
    """
    QBER as a function of distance at a fixed noise level.

    Useful for visualising exactly where the 11 percent threshold is
    crossed and how sharply QBER rises near the security limit. Returns
    both fibre and atmospheric curves so they can be overlaid directly.

    Parameters
    ----------
    noise_std  : float   fixed rotation noise std dev in radians
    Cn2        : float   turbulence strength (None returns fibre only)
    protected  : bool    twirling active
    n_points   : int     number of distance points

    Returns
    -------
    distances   : np.ndarray   distance axis in km
    qber_fibre  : np.ndarray   fibre QBER values in percent
    qber_atmos  : np.ndarray or None   atmospheric QBER (if Cn2 given)
    """
    distances  = np.linspace(0.1, MAX_DIST_SEARCH, n_points)
    qber_fibre = np.array([
        get_qber_fibre(d, noise_std, protected) for d in distances
    ])

    qber_atmos = None
    if Cn2 is not None:
        qber_atmos = np.array([
            get_qber_atmospheric(d, noise_std, Cn2, protected)
            for d in distances
        ])

    return distances, qber_fibre, qber_atmos

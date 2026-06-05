"""
physics/atmosphere.py
---------------------
Atmospheric free-space channel transmittance model.

This module replaces the single fibre line
    trans = 10 ** (-(LOSS_COEFF * distance) / 10)
with a physically grounded three-component model:

    1. Turbulence loss   — Kolmogorov / Fried parameter / Strehl ratio
    2. Diffraction loss  — Gaussian beam spreading with distance
    3. Absorption loss   — molecular extinction at 780 nm

Each component is computed independently and multiplied to give
the total transmittance. This is the Step 1 injection into the
MDI-QKD simulation.

References
----------
Fried, D.L. (1966). Optical resolution through a randomly inhomogeneous
    medium. JOSA 56, 1372.
Marechal, A. (1947). Etude des effets combines de la diffraction et des
    aberrations geometriques. Rev. Opt. 26, 257.
Andrews, L.C. and Phillips, R.L. (2005). Laser Beam Propagation through
    Random Media. SPIE Press.
"""

import warnings
import numpy as np
from atmos_qkd.constants import WAVELENGTH, APERTURE_D, ALPHA_ABS_CLEAR


def get_fried_parameter(Cn2, distance, wavelength=WAVELENGTH):
    """
    Fried coherence length r0 in metres.

    r0 is the diameter of the largest patch of atmosphere that behaves
    like uniform, undistorted glass. When the receiver aperture is
    smaller than r0, turbulence has little effect. When the aperture
    spans many r0 cells, the collected wavefront is severely distorted
    and the Strehl ratio drops rapidly.

    Typical ground-level values:
        r0 ~ 1 to 3 cm    strong turbulence, hot afternoon
        r0 ~ 5 to 10 cm   moderate turbulence, urban daytime
        r0 ~ 20 to 30 cm  weak turbulence, calm night at altitude

    The formula is the Fried (1966) plane-wave approximation. The
    prefactor 0.185 comes from integrating the Kolmogorov turbulence
    power spectrum weighted by the wave propagation kernel along the
    path length L.

    Parameters
    ----------
    Cn2        : float   refractive index structure parameter (m^-2/3)
    distance   : float   propagation path length in km
    wavelength : float   optical wavelength in metres

    Returns
    -------
    r0 : float   Fried parameter in metres
    """
    L = distance * 1000.0
    if L <= 0:
        return 1.0

    r0 = 0.185 * (wavelength ** 2 / (Cn2 * L)) ** (3.0 / 5.0)
    return float(r0)


def get_strehl_ratio(r0, D=APERTURE_D):
    """
    Strehl ratio via the Marechal approximation.

    The Marechal approximation  S ≈ exp(-1.03 × (D/r0)^(5/3))  is
    derived by expanding the focal-plane intensity for small wavefront
    aberrations. It is accurate when D/r0 is not large — typically
    taken as valid for D/r0 ≲ 3. Beyond that the true Strehl requires
    numerical integration of the aberrated pupil function.

    A RuntimeWarning is issued when D/r0 > 3 so that simulation
    results from strong-turbulence regimes are not silently accepted
    as if the approximation still holds. At D/r0 = 10 the Marechal
    value is optimistic by a factor of roughly 2-4× compared to the
    exact Strehl; at D/r0 = 45 (Cn2=1e-13 at 10 km) the link is
    effectively broken and any formula gives numbers near zero.

    Parameters
    ----------
    r0 : float   Fried parameter in metres
    D  : float   receiver aperture diameter in metres

    Returns
    -------
    strehl : float   power fraction in [0, 1]
    """
    if r0 <= 0:
        return 0.0

    ratio = D / r0

    if ratio > 3.0:
        warnings.warn(
            f"Marechal approximation validity exceeded: "
            f"D/r0 = {ratio:.1f}  (reliable range: D/r0 ≤ 3). "
            f"The returned Strehl ratio may be optimistic. "
            f"Consider capping Cn2 at ≈1e-14 for 10 km links.",
            RuntimeWarning,
            stacklevel=2,
        )

    strehl = np.exp(-1.03 * ratio ** (5.0 / 3.0))
    return float(np.clip(strehl, 0.0, 1.0))


def get_geometric_loss(distance, D=APERTURE_D, wavelength=WAVELENGTH):
    """
    Geometric collection efficiency — the fraction of beam power
    captured by the receiver aperture due to diffraction spreading.

    A Gaussian laser beam diverges as it propagates even in a perfect
    vacuum. The beam radius grows from the transmitter waist w0 to
    w(z) at distance z. We assume the transmitter waist is matched
    to the aperture radius (w0 = D/2), which maximises the useful
    range for a given aperture size.

    At short distances the beam has spread little and nearly all
    power is captured. At long distances the beam footprint is much
    wider than the aperture and most power is missed.

    Parameters
    ----------
    distance   : float   propagation distance in km
    D          : float   receiver aperture diameter in metres
    wavelength : float   optical wavelength in metres

    Returns
    -------
    eta_geo : float   collection efficiency in [0, 1]
    """
    L = distance * 1000.0
    if L <= 0:
        return 1.0

    w0  = D / 2.0
    z_R = np.pi * w0 ** 2 / wavelength          # Rayleigh range
    w_z = w0 * np.sqrt(1.0 + (L / z_R) ** 2)   # beam radius at receiver

    # Fraction of Gaussian beam power within a circular aperture
    # of radius D/2, using the standard intensity profile integral.
    eta_geo = 1.0 - np.exp(-2.0 * (D / 2.0) ** 2 / w_z ** 2)
    return float(np.clip(eta_geo, 0.0, 1.0))


def get_absorption_loss(distance, wavelength=WAVELENGTH):
    """
    Molecular absorption transmittance for clear-sky conditions at
    780 nm.

    The primary absorbers at this wavelength are water vapour and
    oxygen. Under clear conditions the extinction coefficient is
    approximately 0.05 dB/km, which is lower than standard fibre
    loss for short links but accumulates over longer paths.

    Fog, haze, and precipitation increase this substantially. Those
    weather-dependent conditions are outside the scope of this
    baseline model but can be incorporated by passing a different
    absorption coefficient.

    Parameters
    ----------
    distance   : float   propagation path length in km
    wavelength : float   wavelength in metres (reserved for future
                         spectral lookup table extension)

    Returns
    -------
    eta_abs : float   absorption transmittance in [0, 1]
    """
    loss_dB = ALPHA_ABS_CLEAR * distance
    return float(10.0 ** (-loss_dB / 10.0))


def get_atmospheric_trans(distance, Cn2,
                          D=APERTURE_D, wavelength=WAVELENGTH):
    """
    Total atmospheric transmittance.

    Combines turbulence, diffraction, and absorption losses by
    multiplication. Each mechanism operates independently on the
    photon beam, so their transmittances multiply to give the overall
    fraction of photons that arrive within the receiver aperture and
    focal spot.

    Parameters
    ----------
    distance   : float   propagation distance in km
    Cn2        : float   turbulence strength in m^-2/3
    D          : float   receiver aperture diameter in metres
    wavelength : float   optical wavelength in metres

    Returns
    -------
    trans : float   total transmittance in [0, 1]
    """
    r0      = get_fried_parameter(Cn2, distance, wavelength)
    strehl  = get_strehl_ratio(r0, D)
    eta_geo = get_geometric_loss(distance, D, wavelength)
    eta_abs = get_absorption_loss(distance, wavelength)

    trans = strehl * eta_geo * eta_abs
    return float(np.clip(trans, 0.0, 1.0))

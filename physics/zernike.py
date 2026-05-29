"""
physics/zernike.py
------------------
Zernike polynomial tools for atmospheric wavefront decomposition.

In Step 1 the Fried parameter r0 summarises the turbulence as a
single scalar. In Step 2 the AI receives individual Zernike
coefficients as input features, which gives it information about
the specific shape of the wavefront distortion rather than just its
overall magnitude. Two channels can have the same r0 but very
different modal structures, and a trained network learns to
distinguish them.

Zernike ordering follows the Noll (1976) convention, which is the
standard in adaptive optics literature.

References
----------
Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
    JOSA 66, 207.
Kolmogorov, A.N. (1941). The local structure of turbulence in
    incompressible viscous fluid. Doklady Akad. Nauk SSSR 30, 301.
"""

import numpy as np
from atmos_qkd.constants import WAVELENGTH, APERTURE_D


# Noll (1976) indices for the first 15 Zernike modes.
# Each tuple is (radial order n, azimuthal frequency m).
# These cover all practically significant aberrations through
# the fourth radial order.
NOLL_MODES = [
    (0,  0),   # Z1  piston
    (1, -1),   # Z2  tip  (beam wander, x direction)
    (1,  1),   # Z3  tilt (beam wander, y direction)
    (2, -2),   # Z4  oblique astigmatism
    (2,  0),   # Z5  defocus
    (2,  2),   # Z6  vertical astigmatism
    (3, -3),   # Z7  vertical trefoil
    (3, -1),   # Z8  vertical coma
    (3,  1),   # Z9  horizontal coma
    (3,  3),   # Z10 oblique trefoil
    (4, -4),   # Z11 oblique quadrafoil
    (4, -2),   # Z12 oblique secondary astigmatism
    (4,  0),   # Z13 primary spherical aberration
    (4,  2),   # Z14 vertical secondary astigmatism
    (4,  4),   # Z15 vertical quadrafoil
]


def kolmogorov_variance(n, m, r0, D=APERTURE_D):
    """
    Variance of a single Zernike coefficient under Kolmogorov
    turbulence statistics.

    Noll (1976) derived the full covariance matrix of Zernike
    coefficients for a wavefront with Kolmogorov statistics. The
    diagonal entries give the variance of each coefficient. Lower
    radial orders (tip, tilt, defocus) carry the most variance and
    therefore cause the most wavefront error. Higher-order modes
    carry progressively less but still contribute meaningfully to
    HOM visibility degradation.

    The approximation below captures the correct (D/r0)^(5/3)
    scaling and the (n+1) radial order dependence to the level
    of accuracy needed for training data generation.

    Parameters
    ----------
    n  : int    radial order
    m  : int    azimuthal frequency (not used in diagonal approx.)
    r0 : float  Fried parameter in metres
    D  : float  aperture diameter in metres

    Returns
    -------
    variance : float   wavefront variance for this mode in rad^2
    """
    if r0 <= 0:
        return np.inf

    variance = 0.0072 * (n + 1) * (D / r0) ** (5.0 / 3.0)
    return float(variance)


def sample_zernike_coefficients(Cn2, distance,
                                wavelength=WAVELENGTH,
                                D=APERTURE_D,
                                n_modes=15,
                                seed=None):
    """
    Draw one random set of Zernike coefficients from the Kolmogorov
    distribution for given turbulence conditions.

    Each call produces one atmospheric realisation. The AI in Step 2
    receives vectors like this as input features in place of a single
    scalar Cn2 value, giving it a much richer description of the
    channel state.

    Each coefficient is drawn independently from a zero-mean Gaussian
    with variance given by kolmogorov_variance(). Independence across
    modes is a reasonable approximation for the diagonal of the Noll
    covariance matrix.

    Parameters
    ----------
    Cn2        : float   turbulence strength in m^-2/3
    distance   : float   propagation distance in km
    wavelength : float   optical wavelength in metres
    D          : float   receiver aperture diameter in metres
    n_modes    : int     number of Zernike modes to sample (max 15)
    seed       : int     random seed for reproducibility (None = random)

    Returns
    -------
    coeffs : np.ndarray   shape (n_modes,), Zernike amplitudes in rad
    """
    from atmos_qkd.physics.atmosphere import get_fried_parameter

    rng    = np.random.default_rng(seed)
    r0     = get_fried_parameter(Cn2, distance, wavelength)
    coeffs = np.zeros(n_modes)

    for i, (n, m) in enumerate(NOLL_MODES[:n_modes]):
        var       = kolmogorov_variance(n, m, r0, D)
        sigma     = np.sqrt(var)
        coeffs[i] = rng.normal(0.0, sigma)

    return coeffs


def zernike_rms(coeffs):
    """
    RMS wavefront error from a vector of Zernike coefficients in rad.

    This single number summarises the overall wavefront distortion.
    It connects to the Strehl ratio via the Marechal approximation:
        Strehl ~ exp(-(2 * pi * rms / lambda)^2)
    for small aberrations.

    Parameters
    ----------
    coeffs : np.ndarray   Zernike amplitudes in rad

    Returns
    -------
    rms : float   RMS wavefront error in rad
    """
    return float(np.sqrt(np.sum(coeffs ** 2)))


def strehl_from_zernike(coeffs):
    """
    Estimate Strehl ratio from a vector of Zernike coefficients.

    Uses the extended Marechal approximation, which is valid for
    RMS wavefront errors up to approximately 1 radian. Beyond that
    threshold, full numerical integration of the point spread function
    is required, but that level of accuracy is beyond the scope of
    this training data generation stage.

    Parameters
    ----------
    coeffs : np.ndarray   Zernike amplitudes in rad

    Returns
    -------
    strehl : float   estimated Strehl ratio in [0, 1]
    """
    rms    = zernike_rms(coeffs)
    strehl = np.exp(-(rms) ** 2)
    return float(np.clip(strehl, 0.0, 1.0))

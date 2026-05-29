"""
qkd/qber.py
-----------
QBER calculation for fibre and atmospheric channels.

The formula structure is identical in both cases:

    QBER = (signal errors + dark count errors) / total detections

The only difference between the two channel types is the
transmittance value passed into that formula. This separation
makes it straightforward to add further channel models later —
any new channel just needs to provide a transmittance, and the
QBER calculation stays unchanged.

The 2/3 factor for the protected case is the exact result of the
correlated twirling protocol derived in Papon et al. (2025).
Averaging over the 12-element unitary 2-design symmetrises the
directional rotation error into an isotropic depolarising channel,
reducing the effective error rate by this constant factor. The
full derivation is in Appendix B of that paper.
"""

import numpy as np
from atmos_qkd.constants import MU, DARK_COUNT_PROB
from atmos_qkd.physics.fibre import get_fibre_trans
from atmos_qkd.physics.atmosphere import get_atmospheric_trans


def _qber_from_trans(trans, noise_std, protected):
    """
    Core QBER formula given a transmittance value.

    This internal function is shared by both the fibre and
    atmospheric calculators. It is not intended to be called
    directly from outside this module.

    Parameters
    ----------
    trans      : float   photon survival probability in [0, 1]
    noise_std  : float   polarization rotation noise std dev in radians
    protected  : bool    whether correlated twirling is active

    Returns
    -------
    qber : float   quantum bit error rate in percent
    """
    p_signal = MU * trans
    p_total  = p_signal + DARK_COUNT_PROB

    # sin^2(theta) is the quantum mechanical probability of measuring
    # the wrong basis state after a polarization rotation of angle theta
    # on the Bloch sphere. This is the intrinsic geometric error from
    # the channel twisting the photon's polarization.
    intr_err = np.sin(noise_std) ** 2

    # The twirling protocol converts the deterministic directional
    # rotation into isotropic depolarising noise. The 2/3 factor is
    # the exact analytical result from Schur's lemma applied to the
    # 12-element unitary 2-design. See Appendix B of Papon et al. (2025).
    if protected:
        intr_err *= (2.0 / 3.0)

    # Dark counts contribute a 50 percent error rate because they are
    # completely uncorrelated with Alice's transmitted bits.
    qber = (p_signal * intr_err + 0.5 * DARK_COUNT_PROB) / p_total
    return float(qber * 100.0)


def get_qber_fibre(distance, noise_std, protected=False):
    """
    QBER for a standard fibre channel.

    Reproduces the Papon et al. (2025) calculation exactly.
    Use this function to validate against the paper and as the
    comparison baseline for all atmospheric results.

    Parameters
    ----------
    distance   : float   fibre length in km
    noise_std  : float   polarization rotation noise std dev in radians
    protected  : bool    correlated twirling active (default False)

    Returns
    -------
    qber : float   QBER in percent
    """
    trans = get_fibre_trans(distance)
    return _qber_from_trans(trans, noise_std, protected)


def get_qber_atmospheric(distance, noise_std, Cn2,
                         protected=False, **kwargs):
    """
    QBER for an atmospheric free-space channel.

    Structurally identical to get_qber_fibre() with the transmittance
    replaced by the Fried/Strehl/diffraction model. This is the
    Step 1 injection point.

    Extra keyword arguments (D, wavelength) are forwarded to
    get_atmospheric_trans() if the default aperture or wavelength
    from constants.py need to be overridden.

    Parameters
    ----------
    distance   : float   free-space path length in km
    noise_std  : float   polarization rotation noise std dev in radians
    Cn2        : float   turbulence strength in m^-2/3
    protected  : bool    correlated twirling active (default False)
    **kwargs             forwarded to get_atmospheric_trans

    Returns
    -------
    qber : float   QBER in percent
    """
    trans = get_atmospheric_trans(distance, Cn2, **kwargs)
    return _qber_from_trans(trans, noise_std, protected)

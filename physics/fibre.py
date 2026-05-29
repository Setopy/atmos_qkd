"""
physics/fibre.py
----------------
Fibre channel transmittance model.

This is the original model from Papon et al. (2025), preserved
unchanged so that the paper's baseline results can always be
reproduced exactly and compared fairly against the atmospheric model.
"""

from atmos_qkd.constants import LOSS_COEFF_FIBRE


def get_fibre_trans(distance):
    """
    Photon survival probability through standard SMF-28 silica fibre.

    Uses the Beer-Lambert law expressed in decibel form. Fibre loss
    of 0.2 dB/km means the signal power decays exponentially with
    distance: 63 percent survive at 10 km, 1 percent at 100 km,
    and 0.01 percent at 150 km. The loss mechanism is primarily
    Rayleigh scattering from frozen-in density fluctuations in the
    glass, with a smaller contribution from material absorption.

    Parameters
    ----------
    distance : float
        Fibre length in km. Must be non-negative.

    Returns
    -------
    trans : float
        Fraction of photons reaching the far end, in [0, 1].
    """
    if distance <= 0:
        return 1.0
    return 10.0 ** (-(LOSS_COEFF_FIBRE * distance) / 10.0)

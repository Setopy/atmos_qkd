"""
constants.py
------------
Single source of truth for every physical and simulation constant
used across the framework.

Changing a value here propagates to all modules automatically.
No magic numbers should appear anywhere else in the codebase.
"""

# ── Optical ──────────────────────────────────────────────────────────
# 780 nm matches the rubidium atomic transition, which is widely used
# in polarization-encoded QKD systems that interface with atomic
# quantum memories. 1550 nm has lower fibre loss but worse single-photon
# detector efficiency, so 780 nm is the more practical choice here.
WAVELENGTH = 780e-9           # metres

# ── Receiver aperture ────────────────────────────────────────────────
# A 10 cm aperture corresponds to a compact portable telescope,
# which is realistic for rooftop or short-range ground experiments.
# Larger apertures capture more photons and tolerate more turbulence
# but increase system weight and cost.
APERTURE_D = 0.10             # metres

# ── Fibre channel ────────────────────────────────────────────────────
# 0.2 dB/km is the combined Rayleigh scattering and material absorption
# limit of standard SMF-28 silica fibre at 1550 nm under laboratory
# conditions. This is the value used in Papon et al. (2025) and is
# kept here to reproduce the paper's baseline results exactly.
LOSS_COEFF_FIBRE = 0.2        # dB/km

# ── Quantum source ───────────────────────────────────────────────────
# Mean photon number for a weak coherent pulse (WCP). Kept below 1
# to limit the photon-number-splitting attack. At MU = 0.1, roughly
# 90 percent of pulses are empty, about 9.5 percent carry one photon,
# and about 0.5 percent carry two or more.
MU = 0.1                      # mean photons per pulse

# ── Detector ─────────────────────────────────────────────────────────
# Probability of a spurious detector click with no photon present.
# Sets the noise floor at long distance: once the signal probability
# drops below this value, QBER climbs toward 50 percent regardless
# of how good the channel is.
DARK_COUNT_PROB = 1e-6        # per pulse

# ── Security threshold ───────────────────────────────────────────────
# Derived from the Shor-Preskill (2000) security proof. The secret
# key rate reaches zero when QBER = 11 percent. Above this value,
# no post-processing procedure can produce a provably secure key.
SECURITY_LIMIT = 11.0         # QBER percent

# ── Turbulence regimes ───────────────────────────────────────────────
# Cn2 is the refractive index structure parameter (m^-2/3). It
# quantifies how strongly temperature fluctuations in the air cause
# refractive index variations along the propagation path.
# These three values represent the practical range from a calm
# high-altitude night to a turbulent hot afternoon at ground level.
CN2_WEAK   = 1e-17            # high altitude, calm night
CN2_MEDIUM = 1e-15            # typical urban ground-level daytime
CN2_STRONG = 1e-13            # heavy turbulence, hot day, sea level

# ── Atmospheric absorption ───────────────────────────────────────────
# Molecular extinction at 780 nm under clear-sky conditions.
# This is significantly lower than fibre loss for short links.
# Fog, haze, and rain increase this substantially and would need
# a separate treatment beyond this baseline model.
ALPHA_ABS_CLEAR = 0.05        # dB/km, clear sky at 780 nm

# ── Simulation resolution ────────────────────────────────────────────
# DIST_STEP = 1 km gives plus or minus 1 km accuracy on maximum
# secure distance, which is sufficient for plotting. For AI training
# data you will want finer steps and more noise samples.
SAMPLES         = 15          # noise levels in the sweep
DIST_STEP       = 1.0         # km per step in the distance search
MAX_DIST_SEARCH = 200         # km upper bound on the distance search

# Noise sweep boundaries in radians.
# 0.01 rad is approximately 0.6 degrees — near-ideal conditions.
# 0.40 rad is approximately 23 degrees — significant channel stress.
NOISE_MIN = 0.01
NOISE_MAX = 0.40

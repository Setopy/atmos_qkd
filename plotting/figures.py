"""
plotting/figures.py
-------------------
All figure-generating functions for the framework.

Each function produces one self-contained plot and optionally saves
it to a file. No physics is computed here — this module only
visualises results that have already been produced by qkd/search.py.

Keeping plotting separate from computation means figures can be
regenerated without re-running the physics, and different data
sources (physics model, AI predictions) can be swapped in without
touching the figure layout code.
"""

import numpy as np
import matplotlib.pyplot as plt
from atmos_qkd.constants import (
    SECURITY_LIMIT, MAX_DIST_SEARCH,
    CN2_WEAK, CN2_MEDIUM, CN2_STRONG,
    WAVELENGTH, APERTURE_D
)
from atmos_qkd.physics.atmosphere import (
    get_atmospheric_trans, get_fried_parameter
)
from atmos_qkd.physics.fibre import get_fibre_trans


def plot_transmittance_comparison(distances=None, save_path=None):
    """
    Fibre vs atmospheric transmittance as a function of distance.

    This figure visualises the injection swap — the original single
    fibre line against the three-component atmospheric model across
    three turbulence regimes.

    Parameters
    ----------
    distances : np.ndarray   distance axis in km (default 0.1 to 200)
    save_path : str          file path to save (None = display only)
    """
    if distances is None:
        distances = np.linspace(0.1, MAX_DIST_SEARCH, 500)

    trans_fibre  = [get_fibre_trans(d)                   for d in distances]
    trans_weak   = [get_atmospheric_trans(d, CN2_WEAK)   for d in distances]
    trans_medium = [get_atmospheric_trans(d, CN2_MEDIUM) for d in distances]
    trans_strong = [get_atmospheric_trans(d, CN2_STRONG) for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(distances, trans_fibre,  'k-',  lw=2.5,
                label='Fibre  0.2 dB/km (paper baseline)')
    ax.semilogy(distances, trans_weak,   'b-',  lw=2,
                label=f'Atmospheric weak    Cn2 = {CN2_WEAK:.0e}')
    ax.semilogy(distances, trans_medium, 'g--', lw=2,
                label=f'Atmospheric medium  Cn2 = {CN2_MEDIUM:.0e}')
    ax.semilogy(distances, trans_strong, 'r:',  lw=2,
                label=f'Atmospheric strong  Cn2 = {CN2_STRONG:.0e}')

    ax.set_xlabel('Distance (km)',             fontsize=12)
    ax.set_ylabel('Transmittance (log scale)', fontsize=12)
    ax.set_title('Step 1 injection: fibre transmittance replaced by '
                 'atmospheric model', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(0, MAX_DIST_SEARCH)
    ax.set_ylim(1e-8, 1.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_fried_parameter(distances=None, save_path=None):
    """
    Fried coherence length r0 vs distance for three turbulence regimes.

    The horizontal line marks the receiver aperture diameter. When r0
    drops below that line, turbulence is resolving multiple coherence
    cells across the aperture and the Strehl ratio falls rapidly.

    Parameters
    ----------
    distances : np.ndarray   distance axis in km
    save_path : str          file path to save
    """
    if distances is None:
        distances = np.linspace(0.1, MAX_DIST_SEARCH, 500)

    r0_weak   = [get_fried_parameter(CN2_WEAK,   d, WAVELENGTH) * 100
                 for d in distances]
    r0_medium = [get_fried_parameter(CN2_MEDIUM, d, WAVELENGTH) * 100
                 for d in distances]
    r0_strong = [get_fried_parameter(CN2_STRONG, d, WAVELENGTH) * 100
                 for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(distances, r0_weak,   'b-',  lw=2, label='Weak turbulence')
    ax.semilogy(distances, r0_medium, 'g--', lw=2, label='Medium turbulence')
    ax.semilogy(distances, r0_strong, 'r:',  lw=2, label='Strong turbulence')
    ax.axhline(y=APERTURE_D * 100, color='purple', lw=1.5,
               linestyle='-.', label=f'Aperture D = {APERTURE_D * 100:.0f} cm')

    ax.set_xlabel('Distance (km)',            fontsize=12)
    ax.set_ylabel('Fried parameter r0 (cm)',  fontsize=12)
    ax.set_title('Fried coherence length vs distance\n'
                 'r0 below aperture D means turbulence dominates',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(0, MAX_DIST_SEARCH)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_figure4_extended(results_fibre, results_atmos_dict,
                          save_path=None):
    """
    Extended version of Figure 4 from Papon et al. (2025).

    Shows the original fibre curves alongside the atmospheric curves
    for each turbulence regime. Dashed lines are unprotected systems;
    solid lines have correlated twirling active.

    Parameters
    ----------
    results_fibre      : dict with keys 'noise', 'raw', 'protected'
    results_atmos_dict : dict of label to result dict, e.g.
                         {'weak': {'noise':..., 'raw':..., 'protected':...}}
    save_path          : str   file path to save
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    noise = results_fibre['noise']
    ax.plot(noise, results_fibre['raw'],       'k--o', ms=5, lw=2,
            label='Fibre unprotected (paper baseline)')
    ax.plot(noise, results_fibre['protected'], 'k-s',  ms=5, lw=2,
            label='Fibre protected (paper baseline)')

    colours = {'weak': 'blue', 'medium': 'green', 'strong': 'red'}
    for label, res in results_atmos_dict.items():
        c = colours.get(label, 'orange')
        ax.plot(res['noise'], res['raw'],
                linestyle='--', marker='o', ms=4, lw=1.5, color=c,
                label=f'Atmospheric {label} unprotected')
        ax.plot(res['noise'], res['protected'],
                linestyle='-',  marker='s', ms=4, lw=1.5, color=c,
                label=f'Atmospheric {label} protected')

    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_xlabel('Noise level — rotation std dev (radians)', fontsize=12)
    ax.set_ylabel('Maximum secure distance (km)',             fontsize=12)
    ax.set_title('Extended Figure 4: fibre vs atmospheric channels\n'
                 'Solid lines = twirling protected   |   '
                 'Dashed lines = unprotected',          fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_ylim(-5, MAX_DIST_SEARCH + 10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_ai_comparison(results_fibre, results_atmos, results_ai,
                       Cn2_label='medium', save_path=None):
    """
    Three-way comparison: fibre baseline, physics model, AI model.

    This is the figure that shows the PhD contribution. The gap
    between the physics model curve and the AI model curve is the
    improvement that the trained network delivers beyond what the
    analytical Fried/Strehl model alone can provide.

    Parameters
    ----------
    results_fibre  : dict   noise, raw, protected — paper baseline
    results_atmos  : dict   noise, raw, protected — physics model
    results_ai     : dict   noise, raw, protected — AI predictions
    Cn2_label      : str    turbulence regime label for the title
    save_path      : str    file path to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(results_fibre['noise'], results_fibre['raw'],
             'k--o', ms=5, lw=2, label='Fibre (paper baseline)')
    ax1.plot(results_atmos['noise'], results_atmos['raw'],
             'g--^', ms=5, lw=2, label=f'Atmospheric physics ({Cn2_label})')
    ax1.plot(results_ai['noise'], results_ai['raw'],
             color='purple', linestyle='--', marker='D', ms=5, lw=2,
             label='AI-predicted atmospheric')
    ax1.axhline(y=0, color='black', lw=0.8)
    ax1.set_title('Unprotected', fontsize=12)
    ax1.set_xlabel('Noise std dev (radians)', fontsize=11)
    ax1.set_ylabel('Maximum secure distance (km)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(-5, MAX_DIST_SEARCH + 10)

    ax2.plot(results_fibre['noise'], results_fibre['protected'],
             'k-s', ms=5, lw=2, label='Fibre protected (paper baseline)')
    ax2.plot(results_atmos['noise'], results_atmos['protected'],
             'g-^', ms=5, lw=2,
             label=f'Atmospheric physics protected ({Cn2_label})')
    ax2.plot(results_ai['noise'], results_ai['protected'],
             color='purple', linestyle='-', marker='D', ms=5, lw=2,
             label='AI-predicted atmospheric protected')
    ax2.axhline(y=0, color='black', lw=0.8)
    ax2.set_title('Protected (correlated twirling active)', fontsize=12)
    ax2.set_xlabel('Noise std dev (radians)', fontsize=11)
    ax2.set_ylabel('Maximum secure distance (km)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(-5, MAX_DIST_SEARCH + 10)

    fig.suptitle(
        'Step 3 result: fibre  →  atmospheric physics  →  AI model\n'
        'Purple above green means the AI improves on the physics-only prediction',
        fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

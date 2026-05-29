"""
run_step3_compare.py
--------------------
Step 3: AI vs physics model comparison.

Loads the trained neural network from Step 2b and uses it to predict
QKD performance across the same noise sweep used in Step 1. Produces
the three-way comparison figure:

    black  — fibre baseline (Papon et al. 2025)
    green  — atmospheric physics model (Fried/Strehl, Step 1)
    purple — AI-predicted atmospheric model (this step)

The gap between the green and purple curves is the AI contribution.
Where purple sits above green, the AI recovers secure range that the
physics-only model could not predict.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step3_compare.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import joblib
except ImportError:
    print("joblib not found. Run:  pip install joblib")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atmos_qkd.constants import (
    CN2_MEDIUM, NOISE_MIN, NOISE_MAX, SAMPLES,
    DIST_STEP, MAX_DIST_SEARCH, SECURITY_LIMIT, MU, DARK_COUNT_PROB
)
from atmos_qkd.qkd.search import sweep_noise_fibre, sweep_noise_atmospheric
from atmos_qkd.physics.zernike import sample_zernike_coefficients
from atmos_qkd.constants import WAVELENGTH, APERTURE_D

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step3")
os.makedirs(OUT_DIR, exist_ok=True)

# Number of Zernike realisations averaged per noise level.
# The AI receives Zernike coefficients as input, but the noise
# sweep axis is polarization rotation noise_std — not a single
# Zernike realisation. We average over N_ZERNIKE_AVG independent
# turbulence realisations to get a stable expected performance
# for each (distance, Cn2, noise_std) combination.
N_ZERNIKE_AVG = 20


def load_model():
    """
    Load the trained network and feature scaler from disk.

    Raises a clear error if run_step2_train.py has not been
    run yet and the model files do not exist.
    """
    model_path  = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run run_step2_train.py first.")
        sys.exit(1)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded from  : {model_path}")
    print(f"Scaler loaded from : {scaler_path}")
    return model, scaler


def ai_predict_qber(distance, noise_std, Cn2, model, scaler,
                    protected, n_avg=N_ZERNIKE_AVG):
    """
    Predict QBER for one (distance, noise_std, Cn2) condition
    using the trained neural network.

    Because the network was trained on individual Zernike
    realisations, we draw n_avg independent wavefront samples
    and average the predictions. This gives a stable expected
    value that is comparable to the deterministic physics model.

    Parameters
    ----------
    distance   : float   path length in km
    noise_std  : float   polarization rotation std dev in radians
    Cn2        : float   turbulence strength in m^-2/3
    model      : trained MLPRegressor
    scaler     : fitted StandardScaler
    protected  : bool    whether to return QBER_protected or QBER_raw
    n_avg      : int     number of Zernike realisations to average

    Returns
    -------
    qber : float   predicted QBER in percent
    """
    rows = []
    for i in range(n_avg):
        coeffs = sample_zernike_coefficients(
            Cn2, distance,
            wavelength=WAVELENGTH,
            D=APERTURE_D,
            n_modes=15,
            seed=i
        )
        row = np.array(
            [distance, Cn2, noise_std] + list(coeffs),
            dtype=np.float64
        ).reshape(1, -1)
        rows.append(row)

    X        = np.vstack(rows)
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)

    # Column indices: 0=trans, 1=QBER_raw, 2=QBER_protected
    col  = 2 if protected else 1
    qber = float(np.mean(y_pred[:, col]))
    return max(0.0, qber)


def find_max_distance_ai(noise_std, Cn2, model, scaler,
                         protected=False):
    """
    Maximum secure distance predicted by the AI model.

    Uses the same linear scan as the physics model so the two
    are directly comparable on the same distance grid.

    Parameters
    ----------
    noise_std : float   polarization rotation std dev in radians
    Cn2       : float   turbulence strength in m^-2/3
    model     : trained MLPRegressor
    scaler    : fitted StandardScaler
    protected : bool    twirling active

    Returns
    -------
    max_dist : float   maximum secure distance in km
    """
    for d in np.arange(0, MAX_DIST_SEARCH, DIST_STEP):
        qber = ai_predict_qber(d, noise_std, Cn2,
                               model, scaler, protected)
        if qber > SECURITY_LIMIT:
            return float(d)
    return float(MAX_DIST_SEARCH)


def sweep_noise_ai(Cn2, model, scaler,
                   n_samples=SAMPLES, protected=False):
    """
    Full noise sweep using AI predictions.

    Parameters
    ----------
    Cn2       : float   turbulence strength in m^-2/3
    model     : trained MLPRegressor
    scaler    : fitted StandardScaler
    n_samples : int     number of noise levels
    protected : bool    twirling active

    Returns
    -------
    noise_levels  : np.ndarray   noise std dev values in radians
    max_distances : np.ndarray   AI-predicted max secure distances in km
    """
    noise_levels = np.linspace(NOISE_MIN, NOISE_MAX, n_samples)
    max_distances = np.array([
        find_max_distance_ai(n, Cn2, model, scaler, protected)
        for n in noise_levels
    ])
    return noise_levels, max_distances


def plot_three_way_comparison(results_fibre, results_physics,
                               results_ai, Cn2_label, save_path=None):
    """
    The primary output figure for Step 3.

    Three curves on two panels (unprotected and protected):
        black  — fibre baseline from the paper
        green  — atmospheric physics model (Step 1)
        purple — AI-predicted atmospheric model (this step)

    Parameters
    ----------
    results_fibre   : dict   noise, raw, protected
    results_physics : dict   noise, raw, protected
    results_ai      : dict   noise, raw, protected
    Cn2_label       : str    turbulence regime label for the title
    save_path       : str    file path to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    noise = results_fibre['noise']

    # ── Left panel: unprotected ──────────────────────────────────
    ax1.plot(noise, results_fibre['raw'],
             'k--o', ms=5, lw=2,
             label='Fibre unprotected (paper baseline)')
    ax1.plot(noise, results_physics['raw'],
             'g--^', ms=5, lw=2,
             label=f'Atmospheric physics unprotected ({Cn2_label})')
    ax1.plot(noise, results_ai['raw'],
             color='purple', linestyle='--', marker='D', ms=5, lw=2,
             label='AI-predicted atmospheric unprotected')

    ax1.axhline(y=0, color='black', lw=0.8)
    ax1.set_title('Unprotected channel', fontsize=13)
    ax1.set_xlabel('Polarization rotation noise std dev (radians)',
                   fontsize=11)
    ax1.set_ylabel('Maximum secure distance (km)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(-5, MAX_DIST_SEARCH + 10)

    # ── Right panel: protected ───────────────────────────────────
    ax2.plot(noise, results_fibre['protected'],
             'k-s', ms=5, lw=2,
             label='Fibre protected (paper baseline)')
    ax2.plot(noise, results_physics['protected'],
             'g-^', ms=5, lw=2,
             label=f'Atmospheric physics protected ({Cn2_label})')
    ax2.plot(noise, results_ai['protected'],
             color='purple', linestyle='-', marker='D', ms=5, lw=2,
             label='AI-predicted atmospheric protected')

    ax2.axhline(y=0, color='black', lw=0.8)
    ax2.set_title('Protected channel (correlated twirling active)',
                  fontsize=13)
    ax2.set_xlabel('Polarization rotation noise std dev (radians)',
                   fontsize=11)
    ax2.set_ylabel('Maximum secure distance (km)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(-5, MAX_DIST_SEARCH + 10)

    fig.suptitle(
        'Step 3: fibre baseline  →  atmospheric physics  →  AI model\n'
        f'Turbulence regime: {Cn2_label}  |  '
        f'QBER security limit: {SECURITY_LIMIT}%',
        fontsize=13
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_qber_vs_distance(noise_std_values, Cn2,
                           model, scaler, save_path=None):
    """
    QBER as a function of distance at selected noise levels.

    Shows physics model (solid) and AI predictions (dashed) on the
    same axes. The horizontal line marks the 11 percent security
    threshold. Where the AI curve crosses that line at a longer
    distance than the physics curve, the AI is predicting a more
    optimistic (better) channel condition.

    Parameters
    ----------
    noise_std_values : list   two or three noise levels to plot
    Cn2              : float  turbulence strength in m^-2/3
    model            : trained MLPRegressor
    scaler           : fitted StandardScaler
    save_path        : str    file path to save
    """
    from atmos_qkd.qkd.search import sweep_distance_qber

    distances = np.linspace(0.5, MAX_DIST_SEARCH, 200)
    colours   = ['steelblue', 'darkorange', 'seagreen']

    fig, ax = plt.subplots(figsize=(11, 6))

    for noise_std, colour in zip(noise_std_values, colours):
        # Physics model
        _, qber_fibre, qber_physics = sweep_distance_qber(
            noise_std, Cn2=Cn2, protected=False, n_points=200)
        ax.plot(distances, qber_physics,
                color=colour, lw=2,
                label=f'Physics  noise={noise_std:.2f} rad')

        # AI predictions
        qber_ai = np.array([
            ai_predict_qber(d, noise_std, Cn2,
                            model, scaler, protected=False)
            for d in distances
        ])
        ax.plot(distances, qber_ai,
                color=colour, lw=2, linestyle='--',
                label=f'AI       noise={noise_std:.2f} rad')

    ax.axhline(y=SECURITY_LIMIT, color='red', lw=1.5,
               linestyle=':', label=f'Security limit {SECURITY_LIMIT}%')
    ax.set_xlabel('Distance (km)',           fontsize=12)
    ax.set_ylabel('QBER (%)',                fontsize=12)
    ax.set_title('QBER vs distance: physics model vs AI\n'
                 'Solid = physics   Dashed = AI prediction',
                 fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 60)
    ax.set_xlim(0, MAX_DIST_SEARCH)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def print_summary(results_fibre, results_physics,
                  results_ai, noise_levels):
    """
    Print a table comparing max secure distance across all three
    models at three representative noise levels.
    """
    check_noise = [0.05, 0.15, 0.30]
    print("\nMax secure distance comparison")
    print("=" * 65)
    print(f"{'Noise':>8}  {'Fibre raw':>10}  "
          f"{'Physics raw':>12}  {'AI raw':>8}  "
          f"{'Fibre prot':>11}  {'Physics prot':>13}  {'AI prot':>9}")
    print("-" * 65)
    for n in check_noise:
        idx = np.argmin(np.abs(noise_levels - n))
        print(
            f"  {n:.2f}  "
            f"{results_fibre['raw'][idx]:>10.0f}  "
            f"{results_physics['raw'][idx]:>12.0f}  "
            f"{results_ai['raw'][idx]:>8.0f}  "
            f"{results_fibre['protected'][idx]:>11.0f}  "
            f"{results_physics['protected'][idx]:>13.0f}  "
            f"{results_ai['protected'][idx]:>9.0f}"
        )
    print("=" * 65)
    print("All distances in km.  Security limit: 11% QBER.")


def main():
    print("Step 3: AI vs physics model comparison")
    print("=" * 45)

    model, scaler = load_model()

    # ── Fibre baseline ────────────────────────────────────────────
    print("\nRunning fibre sweeps (paper baseline)...")
    noise, raw_f  = sweep_noise_fibre(protected=False)
    _,     prot_f = sweep_noise_fibre(protected=True)
    results_fibre = {'noise': noise, 'raw': raw_f, 'protected': prot_f}

    # ── Physics model sweep ───────────────────────────────────────
    print("Running atmospheric physics sweeps...")
    _, raw_p  = sweep_noise_atmospheric(CN2_MEDIUM, protected=False)
    _, prot_p = sweep_noise_atmospheric(CN2_MEDIUM, protected=True)
    results_physics = {'noise': noise, 'raw': raw_p, 'protected': prot_p}

    # ── AI model sweep ────────────────────────────────────────────
    print("Running AI prediction sweeps (this takes a few minutes)...")
    _, raw_ai  = sweep_noise_ai(CN2_MEDIUM, model, scaler,
                                protected=False)
    _, prot_ai = sweep_noise_ai(CN2_MEDIUM, model, scaler,
                                protected=True)
    results_ai = {'noise': noise, 'raw': raw_ai, 'protected': prot_ai}

    # ── Figures ───────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_three_way_comparison(
        results_fibre, results_physics, results_ai,
        Cn2_label='medium (Cn2=1e-15)',
        save_path=os.path.join(OUT_DIR, "fig_three_way_comparison.png"))

    plot_qber_vs_distance(
        noise_std_values=[0.05, 0.15, 0.30],
        Cn2=CN2_MEDIUM,
        model=model,
        scaler=scaler,
        save_path=os.path.join(OUT_DIR, "fig_qber_vs_distance.png"))

    # ── Summary table ─────────────────────────────────────────────
    print_summary(results_fibre, results_physics, results_ai, noise)

    print(f"\nStep 3 complete. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

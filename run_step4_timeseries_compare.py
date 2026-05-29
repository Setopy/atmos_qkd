"""
run_step4_timeseries_compare.py
--------------------------------
Step 4c: time-domain QBER comparison — the key result figure.

Simulates one 24-hour day and computes QBER at every minute under
three protection strategies:

    1. Unprotected        — raw channel, no twirling
    2. Static twirling    — correlated twirling with 2/3 factor
                            (Papon et al. 2025 result)
    3. AI predictive      — TCN predicts next Zernike state and
                            pre-compensates before distortion arrives

The AI advantage shows as reduced time above the 11% security
threshold — less network downtime than static twirling alone.

This is the figure that demonstrates the PhD contribution.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step4_timeseries_compare.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("PyTorch not found. Run:  pip install torch")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atmos_qkd.models.tcn import ZernikeTCN
from atmos_qkd.data.timeseries_generator import (
    generate_cn2_timeseries,
    generate_alpha_timeseries,
    generate_zernike_timeseries,
    STEPS_PER_DAY, NOISE_MIN, NOISE_MAX
)
from atmos_qkd.physics.zernike import strehl_from_zernike
from atmos_qkd.constants import (
    MU, DARK_COUNT_PROB, SECURITY_LIMIT,
    WAVELENGTH, APERTURE_D, CN2_MEDIUM
)
from atmos_qkd.physics.atmosphere import (
    get_atmospheric_trans, get_fried_parameter
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "model_tcn")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step4")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 60    # must match training
N_MODES  = 15
DISTANCE = 10.0  # km


def load_tcn():
    """
    Load the trained TCN and its configuration.

    Returns
    -------
    model : ZernikeTCN  loaded and set to eval mode
    """
    config_path = os.path.join(MODEL_DIR, "tcn_config.txt")
    model_path  = os.path.join(MODEL_DIR, "tcn_final.pt")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run run_step4_timeseries_train.py first.")
        sys.exit(1)

    config = {}
    with open(config_path) as f:
        for line in f:
            k, v = line.strip().split(': ', 1)
            config[k] = int(v)

    model = ZernikeTCN(
        n_features  = config['n_features'],
        n_modes     = config['n_modes'],
        n_channels  = config['n_channels'],
        n_layers    = config['n_layers'],
        kernel_size = config['kernel_size'],
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"TCN loaded from: {model_path}")
    return model


def qber_from_alpha_and_trans(alpha_deg, trans, protected,
                               ai_correction_rad=0.0):
    """
    Compute QBER at one time step given misalignment angle and
    transmittance.

    For the AI case, ai_correction_rad is the predicted rotation
    that the system pre-compensates. The effective noise seen by
    the channel is reduced by this amount.

    Parameters
    ----------
    alpha_deg          : float   current misalignment in degrees
    trans              : float   channel transmittance
    protected          : bool    static twirling active
    ai_correction_rad  : float   predicted rotation correction in rad

    Returns
    -------
    qber : float   QBER in percent
    """
    alpha_rad = np.deg2rad(alpha_deg)

    # Residual rotation after AI pre-compensation
    effective_alpha = alpha_rad - ai_correction_rad
    intr_err = np.sin(effective_alpha / 2.0) ** 2

    if protected:
        intr_err *= (2.0 / 3.0)

    p_signal = MU * trans
    p_total  = p_signal + DARK_COUNT_PROB
    qber     = (p_signal * intr_err + 0.5 * DARK_COUNT_PROB) / p_total
    return float(qber * 100.0)


def predict_next_alpha(model, history_Z, history_cn2,
                        history_alpha, history_noise):
    """
    Use the TCN to predict the next-step Zernike vector and
    convert it to an effective rotation correction.

    The predicted Zernike vector represents what the wavefront
    will look like one step ahead. The tip (Z2) and tilt (Z3)
    coefficients dominate polarization rotation, so we use their
    RMS as the correction to apply.

    Parameters
    ----------
    model          : ZernikeTCN
    history_Z      : np.ndarray  shape (SEQ_LEN, N_MODES)
    history_cn2    : np.ndarray  shape (SEQ_LEN,)
    history_alpha  : np.ndarray  shape (SEQ_LEN,)
    history_noise  : np.ndarray  shape (SEQ_LEN,)

    Returns
    -------
    correction_rad : float   predicted rotation to pre-compensate
    """
    context = np.stack([
        history_cn2,
        np.deg2rad(history_alpha),
        history_noise,
    ], axis=1)   # (SEQ_LEN, 3)

    X = np.concatenate([history_Z, context], axis=1)  # (SEQ_LEN, n_feat)
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_Z = model(X_t).cpu().numpy().squeeze()   # (N_MODES,)

    # Tip and tilt (indices 1 and 2) are the dominant contributors
    # to polarization rotation in the Noll convention.
    correction_rad = float(np.sqrt(pred_Z[1]**2 + pred_Z[2]**2))
    return correction_rad


def run_simulation(seed=99):
    """
    Simulate one 24-hour day and compute QBER at every minute
    for all three protection strategies.

    Parameters
    ----------
    seed : int   random seed for reproducibility

    Returns
    -------
    time         : np.ndarray  (STEPS_PER_DAY,) time axis in minutes
    qber_unprot  : np.ndarray  unprotected QBER
    qber_static  : np.ndarray  static twirling QBER
    qber_ai      : np.ndarray  AI predictive QBER
    """
    model = load_tcn()

    cn2_series    = generate_cn2_timeseries(STEPS_PER_DAY, seed=seed)
    alpha_deg_seq = generate_alpha_timeseries(STEPS_PER_DAY, seed=seed+1)
    zernike_seq   = generate_zernike_timeseries(
        STEPS_PER_DAY, DISTANCE,
        cn2_series=cn2_series, n_modes=N_MODES, seed=seed+2)

    rng = np.random.default_rng(seed)
    noise_std_seq = rng.uniform(NOISE_MIN, NOISE_MAX,
                                 size=STEPS_PER_DAY).astype(np.float32)

    time        = np.arange(STEPS_PER_DAY, dtype=float)
    qber_unprot = np.zeros(STEPS_PER_DAY)
    qber_static = np.zeros(STEPS_PER_DAY)
    qber_ai     = np.zeros(STEPS_PER_DAY)

    print("Running 24-hour simulation...")

    for t in range(STEPS_PER_DAY):
        trans = get_atmospheric_trans(DISTANCE, cn2_series[t])
        alpha = alpha_deg_seq[t]

        qber_unprot[t] = qber_from_alpha_and_trans(
            alpha, trans, protected=False)
        qber_static[t] = qber_from_alpha_and_trans(
            alpha, trans, protected=True)

        # AI prediction — only possible after seq_len steps of history
        if t >= SEQ_LEN:
            correction = predict_next_alpha(
                model,
                zernike_seq[t-SEQ_LEN:t],
                cn2_series[t-SEQ_LEN:t],
                alpha_deg_seq[t-SEQ_LEN:t],
                noise_std_seq[t-SEQ_LEN:t]
            )
            qber_ai[t] = qber_from_alpha_and_trans(
                alpha, trans, protected=True,
                ai_correction_rad=correction)
        else:
            # Before enough history — fall back to static twirling
            qber_ai[t] = qber_static[t]

    return time, qber_unprot, qber_static, qber_ai


def plot_time_domain(time, qber_unprot, qber_static, qber_ai,
                     save_path=None):
    """
    24-hour QBER time series comparing all three protection strategies.

    The filled red region shows unprotected downtime.
    The filled orange region shows residual static-twirling downtime.
    The AI line should have less time above 11% than both.
    """
    # Convert minutes to hours for x-axis labels
    hours = time / 60.0

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(hours, qber_unprot, color='firebrick', lw=1.5, alpha=0.8,
            label='Unprotected')
    ax.plot(hours, qber_static, color='steelblue', lw=1.8,
            label='Static twirling (Papon et al. 2025)')
    ax.plot(hours, qber_ai,     color='purple',    lw=2.0,
            label='AI predictive twirling (this work)')

    ax.axhline(y=SECURITY_LIMIT, color='black', lw=1.5,
               linestyle='--', label=f'Security limit {SECURITY_LIMIT}%')

    # Shade downtime regions
    ax.fill_between(hours, SECURITY_LIMIT, qber_unprot,
                    where=(qber_unprot > SECURITY_LIMIT),
                    color='firebrick', alpha=0.15, interpolate=True,
                    label='Unprotected downtime')
    ax.fill_between(hours, SECURITY_LIMIT, qber_ai,
                    where=(qber_ai > SECURITY_LIMIT),
                    color='purple', alpha=0.2, interpolate=True,
                    label='AI residual downtime')

    ax.set_xlim(0, 24)
    ax.set_ylim(0, min(qber_unprot.max() * 1.1, 60))
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h}h" for h in range(0, 25, 4)])
    ax.set_xlabel('Operational time', fontsize=12)
    ax.set_ylabel('QBER (%)',         fontsize=12)
    ax.set_title(
        '24-hour QBER: unprotected vs static twirling vs AI predictive\n'
        f'Link distance: {DISTANCE} km  |  '
        f'Turbulence: medium (Cn2={CN2_MEDIUM:.0e})',
        fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def print_downtime_summary(qber_unprot, qber_static, qber_ai):
    """
    Print the fraction of time each strategy spends above
    the 11% security threshold.
    """
    T = len(qber_unprot)
    dt_unprot = np.sum(qber_unprot > SECURITY_LIMIT) / T * 100
    dt_static = np.sum(qber_static > SECURITY_LIMIT) / T * 100
    dt_ai     = np.sum(qber_ai     > SECURITY_LIMIT) / T * 100

    print("\nDowntime summary (fraction of 24h above 11% QBER)")
    print("=" * 48)
    print(f"  Unprotected          : {dt_unprot:.1f}%")
    print(f"  Static twirling      : {dt_static:.1f}%")
    print(f"  AI predictive        : {dt_ai:.1f}%")
    print("=" * 48)

    if dt_static > 0:
        reduction = (dt_static - dt_ai) / dt_static * 100
        print(f"  AI downtime reduction vs static: {reduction:.1f}%")


def main():
    print("Step 4c: time-domain comparison")
    print("=" * 45)

    time, qber_unprot, qber_static, qber_ai = run_simulation(seed=99)

    print_downtime_summary(qber_unprot, qber_static, qber_ai)

    plot_time_domain(
        time, qber_unprot, qber_static, qber_ai,
        save_path=os.path.join(OUT_DIR, "fig_timedomain_comparison.png"))

    print(f"\nStep 4c complete. Figures saved to: {OUT_DIR}")
    print("This is the PhD contribution figure.")


if __name__ == "__main__":
    main()

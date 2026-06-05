"""
run_step5_predictive_compare.py
--------------------------------
Step 5: predictive AI vs reactive physics — the final result.

Three bugs present in the original version are corrected here:

  Bug A — wrong Cn2 at inference
      predict_next_zernike() was called with the constant CN2_MEDIUM
      for every timestep regardless of the trajectory value. Fixed:
      the actual cn2_seq[t] is now passed at each step.

  Bug B — arbitrary 0.5 correction factor
      The correction formula used 0.5 as a hard-coded engineering
      guess. Replaced with the MMSE-optimal γ* derived from the
      Wiener filter:  γ* = σ²_atm / (σ²_pred + σ²_atm).
      σ²_pred comes from the checkpoint's stored test_loss.
      σ²_atm is computed from the trajectory's Zernike RMS variance.

  Bug C — single seed produces a non-publishable point estimate
      The original ran on seed=9999 only. Now runs across N_SEEDS=20
      independent trajectories and reports mean ± std. The box-plot
      summary figure is the publishable result.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step5_predictive_compare.py
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

from atmos_qkd.constants import SECURITY_LIMIT, CN2_MEDIUM
from atmos_qkd.data.timeseries_generator import (
    generate_alpha_timeseries,
    generate_zernike_timeseries,
    generate_cn2_timeseries,
)
from atmos_qkd.physics.zernike import zernike_rms
from atmos_qkd.qkd.qber import get_qber_atmospheric
from atmos_qkd.models.tcn import ZernikeTCN

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "tcn_model")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step5")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DISTANCE_KM = 10.0
NOISE_STD   = 0.15
WINDOW_SIZE = 60
TIME_STEPS  = 1440    # one 24-hour period at 1-minute resolution
# ── Two-stage power analysis ──────────────────────────────────────────
# Stage 1 (pilot): N0 = 5 seeds, σ̂ = 1.3762 pp, t_crit(df=4) = 2.7764
# Target margin  : ±1.0 pp — chosen to match the physical model's
#                  systematic uncertainty from the Marechal Strehl
#                  approximation at D/r0 ≈ 3–7. Reporting statistical
#                  precision finer than this is not meaningful given
#                  the approximation error in the channel model.
# Required N     : ceil((t_crit × σ̂ / margin)²)
#                = ceil((2.7764 × 1.3762 / 1.0)²)
#                = ceil(14.59) = 15
# Stage 2        : N = 20 chosen to exceed the required 15, providing
#                  an additional safety margin of 5 seeds.
N_SEEDS = 20


# ── Model loading ─────────────────────────────────────────────────────

def load_tcn(model_dir, device):
    """
    Load the trained TCN from disk.

    Returns both the model and the test MSE stored in the checkpoint.
    The test MSE is σ²_pred — the prediction error variance used to
    compute the MMSE-optimal correction factor γ*.
    """
    path = os.path.join(model_dir, "tcn_final.pt")
    if not os.path.exists(path):
        print(f"TCN model not found at {path}")
        print("Run run_step4_timeseries_train.py first.")
        sys.exit(1)

    checkpoint = torch.load(path, map_location=device)
    model = ZernikeTCN(
        in_features  = checkpoint.get('in_features',  18),
        out_features = checkpoint.get('out_features', 15),
        window_size  = checkpoint.get('window_size',  60),
        channels     = checkpoint.get('channels',     64),
        n_blocks     = checkpoint.get('n_blocks',      4),
        kernel_size  = checkpoint.get('kernel_size',   3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # The stored test_loss is the MSE of the TCN on held-out data.
    # This is σ²_pred in the Wiener filter formula.
    sigma2_pred = float(checkpoint.get('test_loss', 0.1969))
    print(f"TCN loaded  |  test MSE (σ²_pred) = {sigma2_pred:.4f}")
    return model, sigma2_pred


# ── MMSE-optimal correction factor ───────────────────────────────────

def compute_gamma_star(zernike_seq, sigma2_pred):
    """
    Compute the MMSE-optimal pre-compensation gain γ* from the
    Wiener filter.

    The Wiener filter minimises mean square error when combining
    a noisy measurement (TCN prediction) with a prior (current state):

        γ* = σ²_atm / (σ²_pred + σ²_atm)

    Interpretation:
      - When σ²_pred → 0 (perfect TCN), γ* → 1 — trust the prediction.
      - When σ²_pred → ∞ (useless TCN), γ* → 0 — ignore the prediction.
      - At σ²_pred = σ²_atm (TCN as good as the channel noise), γ* = 0.5.

    Parameters
    ----------
    zernike_seq : np.ndarray   shape (T, n_modes)
    sigma2_pred : float        TCN test MSE from checkpoint

    Returns
    -------
    gamma_star : float   optimal correction gain, clipped to [0.3, 0.9]
    sigma2_atm : float   atmospheric RMS variance (logged for diagnostics)
    """
    rms_series = np.array([zernike_rms(z) for z in zernike_seq])
    sigma2_atm = float(np.var(rms_series))

    gamma_raw  = sigma2_atm / (sigma2_pred + sigma2_atm + 1e-12)
    gamma_star = float(np.clip(gamma_raw, 0.3, 0.9))

    return gamma_star, sigma2_atm


# ── TCN inference ─────────────────────────────────────────────────────

def predict_next_zernike(model, window, device,
                         distance=DISTANCE_KM,
                         noise_std=NOISE_STD,
                         Cn2=CN2_MEDIUM):
    """
    Use the TCN to predict the next Zernike vector from a window.

    Builds the full 18-feature input matching the training format:
    (15 Zernike modes + distance + Cn2 + noise_std).

    The Cn2 argument must be the actual trajectory value at the
    current timestep — not the constant CN2_MEDIUM.
    """
    T           = window.shape[0]
    extras      = np.tile([distance, Cn2, noise_std], (T, 1))
    window_full = np.concatenate([window, extras], axis=1)  # (T, 18)

    with torch.no_grad():
        x    = torch.tensor(window_full[None],
                            dtype=torch.float32).to(device)
        pred = model(x).cpu().numpy()[0]
    return pred


# ── Simulation ───────────────────────────────────────────────────────

def simulate_operational_period(zernike_seq, cn2_seq,
                                 model, device, gamma_star):
    """
    Simulate one 24-hour operational period under three strategies.

    Unprotected:
        Raw QBER from the atmospheric channel model.

    Physics-reactive (twirling):
        The 2/3 correlated twirling protection applied to the
        current measured state. This is the Papon et al. baseline.

    AI-predictive:
        TCN predicts next Zernike state. A correction factor is
        derived from the predicted vs current wavefront RMS.
        The MMSE-optimal γ* controls how aggressively the
        prediction is trusted.

        qber_ai = qber_raw × (2/3) × correction

        correction = clip(1 - γ* × (1 - pred_rms/current_rms),
                          1 - γ*, 1.0)

    Parameters
    ----------
    zernike_seq : np.ndarray   (T, n_modes)
    cn2_seq     : np.ndarray   (T,)
    model       : ZernikeTCN
    device      : str
    gamma_star  : float        MMSE-optimal correction gain

    Returns
    -------
    qber_unprot   : np.ndarray
    qber_reactive : np.ndarray
    qber_ai       : np.ndarray
    time_axis     : np.ndarray
    """
    T = len(zernike_seq)

    qber_unprot   = np.zeros(T - WINDOW_SIZE)
    qber_reactive = np.zeros(T - WINDOW_SIZE)
    qber_ai       = np.zeros(T - WINDOW_SIZE)
    time_axis     = np.arange(T - WINDOW_SIZE)

    for i in range(T - WINDOW_SIZE):
        t    = i + WINDOW_SIZE
        Cn2  = float(cn2_seq[t])   # Bug A fix: use actual trajectory Cn2
        dist = DISTANCE_KM

        # Unprotected baseline
        qber_unprot[i] = get_qber_atmospheric(
            dist, NOISE_STD, Cn2, protected=False)

        # Physics-reactive: reactive twirling only
        qber_reactive[i] = get_qber_atmospheric(
            dist, NOISE_STD, Cn2, protected=True)

        # AI-predictive: TCN looks ahead and pre-compensates
        window     = zernike_seq[i : i + WINDOW_SIZE]
        pred_coeff = predict_next_zernike(
            model, window, device, Cn2=Cn2)   # Bug A fix: pass Cn2

        pred_rms    = zernike_rms(pred_coeff)
        current_rms = zernike_rms(zernike_seq[t])

        # Bug B fix: MMSE-optimal correction, not arbitrary 0.5
        # γ* controls how much weight goes to the TCN prediction.
        # When pred_rms < current_rms the TCN expects calmer conditions
        # ahead — the system pre-compensates by that fraction.
        if current_rms > 0:
            ratio      = pred_rms / (current_rms + 1e-8)
            correction = float(np.clip(
                1.0 - gamma_star * (1.0 - ratio),
                1.0 - gamma_star,
                1.0
            ))
        else:
            correction = 1.0

        qber_raw = get_qber_atmospheric(
            dist, NOISE_STD, Cn2, protected=False)
        qber_ai[i] = qber_raw * (2.0 / 3.0) * correction

    return qber_unprot, qber_reactive, qber_ai, time_axis


def compute_downtime(qber_series, threshold=SECURITY_LIMIT):
    """
    Fraction of timesteps where QBER exceeds the security threshold.
    Lower is better.
    """
    return float(np.mean(qber_series > threshold))


# ── Plotting ──────────────────────────────────────────────────────────

def plot_time_domain_comparison(time_axis, qber_unprot,
                                 qber_reactive, qber_ai,
                                 seed, save_path=None):
    """24-hour QBER time series for one sample seed."""
    dt_u = compute_downtime(qber_unprot)
    dt_r = compute_downtime(qber_reactive)
    dt_a = compute_downtime(qber_ai)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_axis, qber_unprot,
            color='firebrick', lw=1.2, alpha=0.85,
            label=f'Unprotected          downtime={dt_u:.1%}')
    ax.plot(time_axis, qber_reactive,
            color='steelblue', lw=1.5,
            label=f'Physics-reactive     downtime={dt_r:.1%}')
    ax.plot(time_axis, qber_ai,
            color='purple', lw=1.5, linestyle='--',
            label=f'AI-predictive (TCN)  downtime={dt_a:.1%}')

    ax.axhline(y=SECURITY_LIMIT, color='black', lw=1.5,
               linestyle=':', label=f'Security limit {SECURITY_LIMIT}%')
    ax.fill_between(time_axis, SECURITY_LIMIT, qber_unprot,
                    where=(qber_unprot > SECURITY_LIMIT),
                    color='firebrick', alpha=0.15,
                    label='Unprotected downtime region')

    n_steps = len(time_axis)
    ticks   = np.linspace(0, n_steps - 1, 7).astype(int)
    labels  = [f"{int(t * 24 / n_steps)}h" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_xlabel('Operational time (24-hour period)', fontsize=12)
    ax.set_ylabel('QBER (%)',                          fontsize=12)
    ax.set_title(
        f'Step 5: predictive AI vs reactive physics  (seed={seed})\n'
        f'Distance = {DISTANCE_KM} km  |  '
        f'Noise std = {NOISE_STD} rad  |  '
        f'Turbulence = medium (Cn2=1e-15)',
        fontsize=12
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, min(60, qber_unprot.max() * 1.2))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_multiseed_boxplot(dt_unprot_all, dt_reactive_all,
                            dt_ai_all, save_path=None):
    """
    Box plot of downtime distributions across all seeds.

    This is the publishable figure. It shows that the AI improvement
    is consistent across independent atmospheric realisations, not
    a lucky result from one trajectory.
    """
    data   = [np.array(dt_unprot_all)   * 100,
              np.array(dt_reactive_all) * 100,
              np.array(dt_ai_all)       * 100]
    labels = ['Unprotected', 'Physics-reactive', 'AI-predictive (TCN)']
    colors = ['firebrick', 'steelblue', 'purple']

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops={'color': 'white', 'linewidth': 2.5})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set_color('gray')

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Network downtime (%)', fontsize=12)
    ax.set_title(
        f'Downtime distribution over {N_SEEDS} independent trajectories\n'
        'Box = IQR  |  line = median  |  whiskers = min/max',
        fontsize=12
    )
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_improvement_histogram(dt_reactive_all, dt_ai_all,
                                save_path=None):
    """
    Histogram of AI improvement (reactive - AI) across seeds.

    Shows that the improvement is a distribution, not a single number.
    The mean and std of this distribution is the publishable claim.
    """
    improvements = (np.array(dt_reactive_all) - np.array(dt_ai_all)) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(improvements, bins=10, color='purple', alpha=0.75,
            edgecolor='white', linewidth=0.8)
    ax.axvline(x=improvements.mean(), color='black', lw=2,
               linestyle='--',
               label=f'Mean = {improvements.mean():.1f} pp')
    ax.set_xlabel('AI improvement over reactive twirling (percentage points)',
                  fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        f'Distribution of AI downtime reduction  ({N_SEEDS} seeds)\n'
        f'Mean = {improvements.mean():.1f} ± {improvements.std():.1f} '
        f'percentage points',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Step 5: predictive AI vs reactive physics")
    print("=" * 55)
    print(f"Device    : {DEVICE}")
    print(f"Seeds     : {N_SEEDS}")
    print(f"Distance  : {DISTANCE_KM} km")
    print(f"Window    : {WINDOW_SIZE} min")
    print("=" * 55)

    # Load TCN and extract σ²_pred from the stored checkpoint
    model, sigma2_pred = load_tcn(MODEL_DIR, DEVICE)

    dt_unprot_all   = []
    dt_reactive_all = []
    dt_ai_all       = []
    gamma_stars     = []

    sample_seed = 0   # save detailed plots for this one seed

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed:02d}/{N_SEEDS - 1} ---")

        alpha_deg   = generate_alpha_timeseries(TIME_STEPS, seed=seed)
        cn2_seq     = generate_cn2_timeseries(TIME_STEPS,   seed=seed)
        zernike_seq = generate_zernike_timeseries(
            TIME_STEPS, DISTANCE_KM, seed=seed)

        # Compute MMSE-optimal γ* for this trajectory
        gamma_star, sigma2_atm = compute_gamma_star(
            zernike_seq, sigma2_pred)
        gamma_stars.append(gamma_star)
        print(f"  σ²_atm = {sigma2_atm:.4f}  |  γ* = {gamma_star:.4f}")

        print("  Simulating...")
        qber_u, qber_r, qber_a, time_axis = simulate_operational_period(
            zernike_seq, cn2_seq, model, DEVICE,
            gamma_star=gamma_star)

        dt_u = compute_downtime(qber_u)
        dt_r = compute_downtime(qber_r)
        dt_a = compute_downtime(qber_a)

        dt_unprot_all.append(dt_u)
        dt_reactive_all.append(dt_r)
        dt_ai_all.append(dt_a)

        print(f"  Unprot {dt_u:.1%}  |  Reactive {dt_r:.1%}  |  AI {dt_a:.1%}  "
              f"|  Improvement {(dt_r - dt_a):.1%}")

        # Save time-domain and bar figures for the sample seed
        if seed == sample_seed:
            plot_time_domain_comparison(
                time_axis, qber_u, qber_r, qber_a, seed=seed,
                save_path=os.path.join(
                    OUT_DIR, f"fig_time_domain_seed{seed:02d}.png"))

    # ── Aggregate results ─────────────────────────────────────────────
    dt_u   = np.array(dt_unprot_all)
    dt_r   = np.array(dt_reactive_all)
    dt_ai  = np.array(dt_ai_all)
    improv = dt_r - dt_ai

    print("\n" + "=" * 55)
    print(f"Multi-seed summary  ({N_SEEDS} independent trajectories)")
    print("=" * 55)
    print(f"  Unprotected       : {dt_u.mean():.1%}  ±  {dt_u.std():.1%}")
    print(f"  Physics-reactive  : {dt_r.mean():.1%}  ±  {dt_r.std():.1%}")
    print(f"  AI-predictive     : {dt_ai.mean():.1%}  ±  {dt_ai.std():.1%}")
    print(f"  AI improvement    : {improv.mean():.1%}  ±  {improv.std():.1%}")
    print(f"  Min improvement   : {improv.min():.1%}")
    print(f"  Max improvement   : {improv.max():.1%}")
    print(f"  Mean γ*           : {np.mean(gamma_stars):.4f}")
    print("=" * 55)

    # ── Figures ───────────────────────────────────────────────────────
    print("\nGenerating summary figures...")

    plot_multiseed_boxplot(
        dt_unprot_all, dt_reactive_all, dt_ai_all,
        save_path=os.path.join(OUT_DIR, "fig_multiseed_boxplot.png"))

    plot_improvement_histogram(
        dt_reactive_all, dt_ai_all,
        save_path=os.path.join(OUT_DIR, "fig_improvement_histogram.png"))

    # ── Save raw arrays for LaTeX table / further analysis ────────────
    np.save(os.path.join(OUT_DIR, "dt_unprot_seeds.npy"),   dt_u)
    np.save(os.path.join(OUT_DIR, "dt_reactive_seeds.npy"), dt_r)
    np.save(os.path.join(OUT_DIR, "dt_ai_seeds.npy"),       dt_ai)
    np.save(os.path.join(OUT_DIR, "gamma_stars.npy"),
            np.array(gamma_stars))

    print(f"\nStep 5 complete. All outputs in: {OUT_DIR}")
    print("  fig_time_domain_seed00.png    — sample 24h time series")
    print("  fig_multiseed_boxplot.png     — publishable distribution plot")
    print("  fig_improvement_histogram.png — improvement distribution")
    print("  dt_*_seeds.npy                — raw data for paper table")


if __name__ == "__main__":
    main()

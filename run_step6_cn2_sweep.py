"""
run_step6_cn2_sweep.py
-----------------------
Step 6: turbulence regime characterisation — the phase diagram.

This script answers the most important scientific question the current
results do not address:

    At what turbulence strength does AI pre-compensation matter most?

It sweeps Cn2 from 1e-17 (calm night) to 1e-14 (strong daytime) and
at each value runs N_SEEDS independent 24-hour simulations, recording
the AI improvement in percentage points of downtime reduction.

The output figure is the phase diagram of system utility. It tells
you and any reviewer:

  — In which turbulence regimes the TCN adds meaningful value
  — Where the improvement saturates (TCN has nothing useful to predict)
  — Where the improvement collapses (turbulence too strong for any correction)

This is the figure that transforms the work from a single operating
point into a characterisation of a system.

NOTE: CN2_STRONG = 1e-13 at 10 km gives D/r0 ≈ 45, well outside the
Marechal approximation's validity range. The sweep is capped at 1e-14
(D/r0 ≈ 10) to keep results physically meaningful. Results above
1e-14 should be treated as indicative only.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step6_cn2_sweep.py
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
    generate_cn2_timeseries,
)
from atmos_qkd.physics.atmosphere import (
    get_fried_parameter, get_atmospheric_trans
)
from atmos_qkd.physics.zernike import (
    zernike_rms, sample_zernike_coefficients
)
from atmos_qkd.qkd.qber import get_qber_atmospheric
from atmos_qkd.models.tcn import ZernikeTCN
from atmos_qkd.constants import WAVELENGTH, APERTURE_D

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "tcn_model")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step7")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DISTANCE_KM = 10.0
NOISE_STD   = 0.15
WINDOW_SIZE = 60
TIME_STEPS  = 1440
N_SEEDS     = 10    # seeds per Cn2 level

# Turbulence sweep: 12 points from 1e-17 to 1e-14
# Cap at 1e-14 to stay within Marechal approximation validity.
CN2_VALUES = np.logspace(-17, -14, 12)


# ── Helper: generate a Zernike time series at fixed Cn2 ──────────────

def generate_zernike_timeseries_fixed_cn2(n_steps, distance, cn2,
                                           n_modes=15, seed=None):
    """
    Generate a Zernike coefficient time series with fixed Cn2.

    Unlike the standard timeseries_generator which uses a drifting
    Cn2 trajectory, this function keeps Cn2 constant so the sweep
    isolates the effect of turbulence strength cleanly.

    Each step draws a fresh set of Kolmogorov Zernike coefficients
    parameterised by the fixed Cn2.
    """
    rng = np.random.default_rng(seed)
    seq = np.zeros((n_steps, n_modes), dtype=np.float32)

    for t in range(n_steps):
        # Each timestep is an independent Kolmogorov realisation.
        # The temporal correlation in real turbulence is handled by
        # the TCN window — the training data already embeds it.
        seed_t = int(rng.integers(0, 2**31))
        seq[t] = sample_zernike_coefficients(
            cn2, distance, n_modes=n_modes, seed=seed_t
        ).astype(np.float32)

    return seq


# ── Model loading ─────────────────────────────────────────────────────

def load_tcn(model_dir, device):
    """Load TCN and return model + σ²_pred."""
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

    sigma2_pred = float(checkpoint.get('test_loss', 0.1969))
    print(f"TCN loaded  |  σ²_pred = {sigma2_pred:.4f}")
    return model, sigma2_pred


# ── MMSE correction ───────────────────────────────────────────────────

def compute_gamma_star(zernike_seq, sigma2_pred):
    """MMSE-optimal correction factor: γ* = σ²_atm / (σ²_pred + σ²_atm)."""
    rms_series = np.array([zernike_rms(z) for z in zernike_seq])
    sigma2_atm = float(np.var(rms_series))
    gamma_raw  = sigma2_atm / (sigma2_pred + sigma2_atm + 1e-12)
    return float(np.clip(gamma_raw, 0.3, 0.9))


# ── TCN inference ─────────────────────────────────────────────────────

def predict_next_zernike(model, window, device, Cn2):
    """Predict next Zernike vector given a window of past states."""
    T           = window.shape[0]
    extras      = np.tile([DISTANCE_KM, Cn2, NOISE_STD], (T, 1))
    window_full = np.concatenate([window, extras], axis=1)

    with torch.no_grad():
        x    = torch.tensor(window_full[None],
                            dtype=torch.float32).to(device)
        pred = model(x).cpu().numpy()[0]
    return pred


# ── One Cn2 level, one seed ───────────────────────────────────────────

def run_one_seed(cn2, seed, model, sigma2_pred, device):
    """
    Run a full 24-hour simulation at one fixed Cn2 and one seed.

    Returns the downtime fractions for all three strategies.
    """
    zernike_seq = generate_zernike_timeseries_fixed_cn2(
        TIME_STEPS, DISTANCE_KM, cn2, n_modes=15, seed=seed)

    gamma_star = compute_gamma_star(zernike_seq, sigma2_pred)

    qber_u = np.zeros(TIME_STEPS - WINDOW_SIZE)
    qber_r = np.zeros(TIME_STEPS - WINDOW_SIZE)
    qber_a = np.zeros(TIME_STEPS - WINDOW_SIZE)

    for i in range(TIME_STEPS - WINDOW_SIZE):
        t = i + WINDOW_SIZE

        qber_u[i] = get_qber_atmospheric(
            DISTANCE_KM, NOISE_STD, cn2, protected=False)
        qber_r[i] = get_qber_atmospheric(
            DISTANCE_KM, NOISE_STD, cn2, protected=True)

        window     = zernike_seq[i : i + WINDOW_SIZE]
        pred_coeff = predict_next_zernike(model, window, device, cn2)

        pred_rms    = zernike_rms(pred_coeff)
        current_rms = zernike_rms(zernike_seq[t])

        if current_rms > 0:
            ratio      = pred_rms / (current_rms + 1e-8)
            correction = float(np.clip(
                1.0 - gamma_star * (1.0 - ratio),
                1.0 - gamma_star, 1.0
            ))
        else:
            correction = 1.0

        qber_raw  = get_qber_atmospheric(
            DISTANCE_KM, NOISE_STD, cn2, protected=False)
        qber_a[i] = qber_raw * (2.0 / 3.0) * correction

    dt_u = float(np.mean(qber_u > SECURITY_LIMIT))
    dt_r = float(np.mean(qber_r > SECURITY_LIMIT))
    dt_a = float(np.mean(qber_a > SECURITY_LIMIT))
    return dt_u, dt_r, dt_a


# ── Sweep ─────────────────────────────────────────────────────────────

def run_sweep(model, sigma2_pred, device):
    """
    Run the full Cn2 sweep.

    Returns arrays of shape (n_cn2,) for means and stds.
    """
    n = len(CN2_VALUES)

    mean_u = np.zeros(n)
    mean_r = np.zeros(n)
    mean_a = np.zeros(n)
    std_u  = np.zeros(n)
    std_r  = np.zeros(n)
    std_a  = np.zeros(n)

    for j, cn2 in enumerate(CN2_VALUES):
        r0     = get_fried_parameter(cn2, DISTANCE_KM)
        ratio  = APERTURE_D / r0
        print(f"\n[{j+1:02d}/{n}]  Cn2={cn2:.1e}  "
              f"r0={r0*100:.1f} cm  D/r0={ratio:.1f}")

        dtu_s, dtr_s, dta_s = [], [], []
        for seed in range(N_SEEDS):
            du, dr, da = run_one_seed(
                cn2, seed, model, sigma2_pred, device)
            dtu_s.append(du)
            dtr_s.append(dr)
            dta_s.append(da)
            print(f"  seed {seed}: U={du:.1%}  R={dr:.1%}  AI={da:.1%}  "
                  f"improv={dr - da:.1%}")

        mean_u[j], std_u[j] = np.mean(dtu_s), np.std(dtu_s)
        mean_r[j], std_r[j] = np.mean(dtr_s), np.std(dtr_s)
        mean_a[j], std_a[j] = np.mean(dta_s), np.std(dta_s)

    return mean_u, std_u, mean_r, std_r, mean_a, std_a


# ── Plotting ──────────────────────────────────────────────────────────

def plot_cn2_sweep(mean_u, std_u, mean_r, std_r,
                   mean_a, std_a, save_path=None):
    """
    Phase diagram: downtime vs turbulence strength.

    X-axis: log10(Cn2) — turbulence intensity
    Y-axis: mean downtime fraction (%) with ± std shading

    The gap between the reactive (blue) and AI (purple) curves is
    the quantified benefit of the TCN pre-compensation as a function
    of operating regime.
    """
    x = np.log10(CN2_VALUES)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10),
                                    sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── Top panel: absolute downtime ──────────────────────────────────
    ax1.plot(x, mean_u * 100, color='firebrick', lw=2,
             label='Unprotected')
    ax1.fill_between(x,
                     (mean_u - std_u) * 100,
                     (mean_u + std_u) * 100,
                     color='firebrick', alpha=0.15)

    ax1.plot(x, mean_r * 100, color='steelblue', lw=2,
             label='Physics-reactive (twirling)')
    ax1.fill_between(x,
                     (mean_r - std_r) * 100,
                     (mean_r + std_r) * 100,
                     color='steelblue', alpha=0.15)

    ax1.plot(x, mean_a * 100, color='purple', lw=2, linestyle='--',
             label='AI-predictive (TCN)')
    ax1.fill_between(x,
                     (mean_a - std_a) * 100,
                     (mean_a + std_a) * 100,
                     color='purple', alpha=0.15)

    ax1.axhline(y=0, color='black', lw=0.8, linestyle=':')
    ax1.set_ylabel('Network downtime (%)', fontsize=12)
    ax1.set_title(
        'Turbulence regime characterisation\n'
        f'Distance = {DISTANCE_KM} km  |  '
        f'{N_SEEDS} seeds per Cn2 level  |  mean ± std shading',
        fontsize=12
    )
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Marechal validity boundary
    # At D/r0 = 3: Cn2 ≈ 3.2e-15 for 10 km at 780 nm
    cn2_validity = 3.2e-15
    ax1.axvline(x=np.log10(cn2_validity), color='gray',
                lw=1.2, linestyle=':',
                label='Marechal validity limit (D/r0 = 3)')

    # ── Bottom panel: AI improvement over reactive ────────────────────
    improv_mean = (mean_r - mean_a) * 100
    improv_std  = np.sqrt(std_r**2 + std_a**2) * 100  # propagated std

    ax2.plot(x, improv_mean, color='purple', lw=2.5,
             label='AI improvement over reactive (pp)')
    ax2.fill_between(x,
                     improv_mean - improv_std,
                     improv_mean + improv_std,
                     color='purple', alpha=0.2)
    ax2.axhline(y=0, color='black', lw=1.2, linestyle='--')
    ax2.axvline(x=np.log10(cn2_validity), color='gray',
                lw=1.2, linestyle=':',
                label='Marechal validity limit')

    ax2.set_xlabel('Turbulence strength  log₁₀(Cn²)  [m⁻²/³]', fontsize=12)
    ax2.set_ylabel('AI improvement (percentage points)', fontsize=12)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Label the Cn2 axis with physical interpretations
    cn2_ticks = [-17, -16, -15, -14]
    labels_cn2 = [
        'Calm night\n(1e-17)',
        'Weak\n(1e-16)',
        'Moderate\n(1e-15)',
        'Strong\n(1e-14)',
    ]
    ax2.set_xticks(cn2_ticks)
    ax2.set_xticklabels(labels_cn2, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def print_sweep_table(mean_u, std_u, mean_r, std_r, mean_a, std_a):
    """Print a LaTeX-ready summary table of the sweep results."""
    print("\n" + "=" * 75)
    print(f"{'Cn2':>10}  {'D/r0':>5}  "
          f"{'Unprot (%)':>12}  {'Reactive (%)':>13}  "
          f"{'AI (%)':>10}  {'Improv (pp)':>12}")
    print("-" * 75)
    for j, cn2 in enumerate(CN2_VALUES):
        r0    = get_fried_parameter(cn2, DISTANCE_KM)
        ratio = APERTURE_D / r0
        improv = (mean_r[j] - mean_a[j]) * 100
        print(f"{cn2:>10.1e}  {ratio:>5.1f}  "
              f"{mean_u[j]*100:>8.1f}±{std_u[j]*100:<3.1f}  "
              f"{mean_r[j]*100:>9.1f}±{std_r[j]*100:<3.1f}  "
              f"{mean_a[j]*100:>6.1f}±{std_a[j]*100:<3.1f}  "
              f"{improv:>+10.1f}")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Step 6: turbulence regime characterisation (Cn2 sweep)")
    print("=" * 55)
    print(f"Device      : {DEVICE}")
    print(f"Cn2 values  : {len(CN2_VALUES)} levels from 1e-17 to 1e-14")
    print(f"Seeds/level : {N_SEEDS}")
    print(f"Total runs  : {len(CN2_VALUES) * N_SEEDS}")
    print("=" * 55)

    model, sigma2_pred = load_tcn(MODEL_DIR, DEVICE)

    mean_u, std_u, mean_r, std_r, mean_a, std_a = run_sweep(
        model, sigma2_pred, DEVICE)

    print_sweep_table(mean_u, std_u, mean_r, std_r, mean_a, std_a)

    # Save arrays for further analysis
    np.save(os.path.join(OUT_DIR, "cn2_values.npy"),  CN2_VALUES)
    np.save(os.path.join(OUT_DIR, "mean_u.npy"),       mean_u)
    np.save(os.path.join(OUT_DIR, "std_u.npy"),        std_u)
    np.save(os.path.join(OUT_DIR, "mean_r.npy"),       mean_r)
    np.save(os.path.join(OUT_DIR, "std_r.npy"),        std_r)
    np.save(os.path.join(OUT_DIR, "mean_a.npy"),       mean_a)
    np.save(os.path.join(OUT_DIR, "std_a.npy"),        std_a)

    # Generate the phase diagram
    print("\nGenerating phase diagram figure...")
    plot_cn2_sweep(
        mean_u, std_u, mean_r, std_r, mean_a, std_a,
        save_path=os.path.join(OUT_DIR, "fig_cn2_phase_diagram.png"))

    print(f"\nStep 7 complete. Outputs in: {OUT_DIR}")
    print("  fig_cn2_phase_diagram.png  — the phase diagram figure")
    print("  *.npy                      — raw arrays for further analysis")


if __name__ == "__main__":
    main()

"""
run_step1.py
------------
Step 1 entry point — atmospheric physics injection.

All physics, search logic, and plotting live in their respective
modules. This script connects them, runs the sweeps, saves the
figures, and prints a summary table.

Run from the directory containing atmos_qkd/:
    python atmos_qkd/run_step1.py
"""

import numpy as np
import os
from atmos_qkd.constants import CN2_WEAK, CN2_MEDIUM, CN2_STRONG
from atmos_qkd.qkd.search import sweep_noise_fibre, sweep_noise_atmospheric
from atmos_qkd.plotting.figures import (
    plot_transmittance_comparison,
    plot_fried_parameter,
    plot_figure4_extended,
)

OUTPUT_DIR = "outputs/step1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Step 1: atmospheric physics injection")
    print("=" * 45)

    # Fibre baseline — reproduces the paper exactly
    print("Running fibre sweeps...")
    noise, dist_raw  = sweep_noise_fibre(protected=False)
    _,     dist_prot = sweep_noise_fibre(protected=True)
    results_fibre = {
        'noise'     : noise,
        'raw'       : dist_raw,
        'protected' : dist_prot,
    }

    # Atmospheric sweeps across three turbulence regimes
    print("Running atmospheric sweeps (weak turbulence)...")
    _, raw_w  = sweep_noise_atmospheric(CN2_WEAK,   protected=False)
    _, prot_w = sweep_noise_atmospheric(CN2_WEAK,   protected=True)

    print("Running atmospheric sweeps (medium turbulence)...")
    _, raw_m  = sweep_noise_atmospheric(CN2_MEDIUM, protected=False)
    _, prot_m = sweep_noise_atmospheric(CN2_MEDIUM, protected=True)

    print("Running atmospheric sweeps (strong turbulence)...")
    _, raw_s  = sweep_noise_atmospheric(CN2_STRONG, protected=False)
    _, prot_s = sweep_noise_atmospheric(CN2_STRONG, protected=True)

    results_atmos = {
        'weak'  : {'noise': noise, 'raw': raw_w,  'protected': prot_w},
        'medium': {'noise': noise, 'raw': raw_m,  'protected': prot_m},
        'strong': {'noise': noise, 'raw': raw_s,  'protected': prot_s},
    }

    # Generate and save figures
    print("\nGenerating figures...")
    plot_transmittance_comparison(
        save_path=f"{OUTPUT_DIR}/fig_trans_comparison.png")

    plot_fried_parameter(
        save_path=f"{OUTPUT_DIR}/fig_fried_parameter.png")

    plot_figure4_extended(
        results_fibre, results_atmos,
        save_path=f"{OUTPUT_DIR}/fig_figure4_extended.png")

    # Summary table at noise = 0.10 rad
    print("\n" + "=" * 45)
    print("Maximum secure distance at noise = 0.10 rad")
    print("=" * 45)
    idx = np.argmin(np.abs(noise - 0.10))
    rows = [
        ("Fibre         unprotected", dist_raw[idx]),
        ("Fibre         protected  ", dist_prot[idx]),
        ("Atmospheric weak   unprot", raw_w[idx]),
        ("Atmospheric weak   prot  ", prot_w[idx]),
        ("Atmospheric medium unprot", raw_m[idx]),
        ("Atmospheric medium prot  ", prot_m[idx]),
        ("Atmospheric strong unprot", raw_s[idx]),
        ("Atmospheric strong prot  ", prot_s[idx]),
    ]
    for label, val in rows:
        print(f"  {label} : {val:.0f} km")
    print("=" * 45)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
run_step4_timeseries_generate.py
---------------------------------
Step 4a: generate the time-series training dataset.

Simulates n_days independent 24-hour atmospheric sequences and
slices them into overlapping (input_sequence, next_step_target)
pairs for the TCN.

Output:
    atmos_qkd/data/timeseries_X.npy
    atmos_qkd/data/timeseries_y.npy
    atmos_qkd/data/timeseries_meta.txt

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step4_timeseries_generate.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atmos_qkd.data.timeseries_generator import (
    generate_timeseries_dataset,
    save_timeseries_dataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "timeseries")

N_DAYS   = 100
SEQ_LEN  = 60
DISTANCE = 10.0
N_MODES  = 15


def main():
    print("Step 4a: time-series dataset generation")
    print("=" * 48)
    print(f"Days to simulate : {N_DAYS}")
    print(f"Sequence length  : {SEQ_LEN} minutes")
    print(f"Link distance    : {DISTANCE} km")
    print(f"Zernike modes    : {N_MODES}")
    print("=" * 48)

    X, y, meta = generate_timeseries_dataset(
        n_days=N_DAYS,
        distance=DISTANCE,
        seq_len=SEQ_LEN,
        n_modes=N_MODES,
        seed=42,
    )

    print(f"\nDataset shape")
    print(f"  X : {X.shape}  (examples, seq_len, features)")
    print(f"  y : {y.shape}  (examples, n_modes)")

    save_timeseries_dataset(X, y, meta, OUT_PATH)
    print("\nStep 4a complete. Run run_step4_timeseries_train.py next.")


if __name__ == "__main__":
    main()

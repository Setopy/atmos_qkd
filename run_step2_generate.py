"""
run_step2_generate.py
---------------------
Step 2a: training data generation.

Samples the atmospheric parameter space and records the physical
inputs alongside the computed QKD performance outputs. The resulting
CSV file is what the neural network trains on in run_step2_train.py.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step2_generate.py

The script will create:
    atmos_qkd/data/training_data.csv
"""

import sys
import os

# Add the parent directory to the path so atmos_qkd is importable
# regardless of where Python looks by default on this machine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atmos_qkd.data.generator import generate_dataset, save_dataset_csv

# Output path — sits inside the package so it travels with the code
OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "training_data.csv"
)

# Number of samples. 200000
N_SAMPLES = 200000


def main():
    print("Step 2a: generating atmospheric QKD training dataset")
    print("=" * 52)
    print(f"Samples     : {N_SAMPLES}")
    print(f"Output file : {OUTPUT_CSV}")
    print("=" * 52)

    data = generate_dataset(n_samples=N_SAMPLES, seed=42)
    save_dataset_csv(data, OUTPUT_CSV)

    print("\nSample row (first entry):")
    for key, val in list(data[0].items())[:8]:
        print(f"  {key:20s} : {val:.6g}")
    print("  ...")
    print("\nStep 2a complete. Run run_step2_train.py next.")


if __name__ == "__main__":
    main()

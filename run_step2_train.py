"""
run_step2_train.py
------------------
Step 2b: train the neural network on the generated dataset.

Reads the CSV produced by run_step2_generate.py and trains a
feedforward network to predict atmospheric transmittance from
the physical input features. The trained model is saved to disk
and can be loaded by run_step3_compare.py to produce the final
AI vs physics comparison figures.

Input features (18 total):
    distance, Cn2, noise_std, z_00 through z_14 (Zernike coefficients)

Targets (3 total):
    trans, QBER_raw, QBER_protected

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step2_train.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
except ImportError:
    print("pandas not found. Run:  pip install pandas")
    sys.exit(1)

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    print("scikit-learn not found. Run:  pip install scikit-learn")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("joblib not found. Run:  pip install joblib")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step2")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

FEATURE_COLS = (
    ["distance", "Cn2", "noise_std"] +
    [f"z_{i:02d}" for i in range(15)]
)
TARGET_COLS = ["trans", "QBER_raw", "QBER_protected"]


def load_and_clean(csv_path):
    df     = pd.read_csv(csv_path)
    before = len(df)
    df     = df[df["trans"] > 1e-7]
    after  = len(df)
    print(f"Loaded {before} rows. Kept {after} after removing "
          f"zero-transmittance entries.")
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COLS].values.astype(np.float64)
    return X, y


def build_and_train(X_train, y_train):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model    = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    print("Training neural network...")
    model.fit(X_scaled, y_train)
    print(f"Stopped at iteration {model.n_iter_}")
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    print("\nTest set performance")
    print("=" * 50)
    for i, name in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2  = r2_score(y_test[:, i], y_pred[:, i])
        print(f"  {name:20s}  MAE = {mae:.4f}   R2 = {r2:.4f}")
    print("=" * 50)
    return y_pred


def plot_predictions(y_test, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        ax.scatter(y_test[:, i], y_pred[:, i],
                   alpha=0.3, s=8, color='steelblue')
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5,
                label='Perfect prediction')
        ax.set_xlabel(f'Actual {name}',    fontsize=11)
        ax.set_ylabel(f'Predicted {name}', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle('Neural network: predicted vs actual values',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_loss_curve(model):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(model.loss_curve_, color='steelblue', lw=2,
            label='Training loss')
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        val_loss = [1 - s for s in model.validation_scores_]
        ax.plot(val_loss, color='firebrick', lw=2,
                linestyle='--', label='Validation loss (1 - R2)')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss',      fontsize=11)
    ax.set_title('Training curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def main():
    print("Step 2b: neural network training")
    print("=" * 45)

    X, y = load_and_clean(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Input features   : {X_train.shape[1]}")
    print(f"Output targets   : {y_train.shape[1]}")

    model, scaler = build_and_train(X_train, y_train)

    y_pred = evaluate(model, scaler, X_test, y_test)

    joblib.dump(model,  os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nModel saved  : {MODEL_DIR}/model.pkl")
    print(f"Scaler saved : {MODEL_DIR}/scaler.pkl")

    plot_predictions(y_test, y_pred)
    plot_loss_curve(model)

    print("\nStep 2b complete. Run run_step3_compare.py next.")


if __name__ == "__main__":
    main()

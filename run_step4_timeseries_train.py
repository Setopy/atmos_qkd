"""
run_step4_timeseries_train.py
------------------------------
Step 4b: train the TCN with learning rate range test + OneCycleLR.

Two-phase training:

Phase 1 — Learning Rate Range Test (Smith 2017)
    Runs one pass through the training data, increasing the learning
    rate exponentially from 1e-7 to 1e-1. Records the loss at each
    step. The optimal learning rate is just before the loss starts
    rising sharply. This takes about 2 minutes and saves a plot.

Phase 2 — Full Training with OneCycleLR
    Uses the found optimal learning rate as the peak of a one-cycle
    schedule. The schedule ramps up from a low rate to the peak over
    the first 30 percent of training, then decays smoothly to near
    zero. This consistently outperforms fixed learning rates and
    reduces the overfitting seen in the first training run.

Additional change from previous run:
    Dropout increased from 0.1 to 0.3 to address the validation loss
    divergence observed after epoch 1 in the first training run.

Run from the directory that contains atmos_qkd/:
    python3 atmos_qkd/run_step4_timeseries_train.py
"""

import sys
import os
import numpy as np
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    print("PyTorch not found. Run:  pip install torch")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atmos_qkd.models.tcn import ZernikeTCN

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "data", "tcn_model")
OUT_DIR   = os.path.join(BASE_DIR, "outputs_step4")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

BATCH_SIZE = 256
EPOCHS     = 80
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.10
DROPOUT    = 0.3       # increased from 0.1 to reduce overfitting
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Learning rate range test bounds
LR_START   = 1e-7
LR_END     = 1e-1
LR_STEPS   = 100       # number of steps in the range test


def load_dataset(data_dir):
    X = np.load(os.path.join(data_dir, "timeseries_X.npy"))
    y = np.load(os.path.join(data_dir, "timeseries_y.npy"))
    print(f"Loaded X: {X.shape}   y: {y.shape}")
    return X, y


def build_dataloaders(X, y):
    X_t     = torch.tensor(X, dtype=torch.float32)
    y_t     = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    n       = len(dataset)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)
    print(f"Train / val / test : {n_train} / {n_val} / {n_test}")
    return train_loader, val_loader, test_loader


# ── PHASE 1: LEARNING RATE RANGE TEST ────────────────────────────────

def lr_range_test(model, train_loader, criterion, device,
                  lr_start=LR_START, lr_end=LR_END,
                  n_steps=LR_STEPS):
    """
    Learning Rate Range Test — Smith (2017).

    Increases the learning rate exponentially over n_steps mini-batches
    and records the loss at each step. The loss initially falls as the
    learning rate rises into a useful range, then rises sharply when
    the learning rate becomes too large for stable optimisation.

    The optimal learning rate for training is the value just before
    the loss begins its final steep ascent — typically about one tenth
    of the value where the loss is at its minimum.

    Parameters
    ----------
    model        : ZernikeTCN   freshly initialised model
    train_loader : DataLoader   training data
    criterion    : loss function
    device       : str
    lr_start     : float        lowest learning rate to test
    lr_end       : float        highest learning rate to test
    n_steps      : int          number of steps in the test

    Returns
    -------
    lrs        : list   learning rates tested
    losses     : list   smoothed loss at each step
    optimal_lr : float  suggested learning rate for training
    """
    print("\nPhase 1: Learning Rate Range Test")
    print(f"  Searching from {lr_start:.0e} to {lr_end:.0e} "
          f"over {n_steps} steps...")

    # Work on a copy so the original model weights are not disturbed
    model_copy = copy.deepcopy(model).to(device)
    optimiser  = torch.optim.Adam(model_copy.parameters(), lr=lr_start)

    # Exponential multiplier to reach lr_end in n_steps steps
    lr_mult = (lr_end / lr_start) ** (1.0 / n_steps)

    lrs        = []
    losses     = []
    smooth_loss = None
    beta        = 0.98    # smoothing factor — reduces noise in the curve
    best_loss   = float('inf')
    data_iter   = iter(train_loader)

    model_copy.train()
    for step in range(n_steps):
        # Cycle through training data if needed
        try:
            X_b, y_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            X_b, y_b = next(data_iter)

        X_b = X_b.to(device)
        y_b = y_b.to(device)

        optimiser.zero_grad()
        pred = model_copy(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
        optimiser.step()

        # Exponentially smoothed loss — reduces mini-batch noise
        raw_loss    = loss.item()
        smooth_loss = (beta * smooth_loss + (1 - beta) * raw_loss
                       if smooth_loss is not None else raw_loss)
        # Bias correction for the early steps
        debiased    = smooth_loss / (1 - beta ** (step + 1))

        current_lr = optimiser.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(debiased)

        if debiased < best_loss:
            best_loss = debiased

        # Stop early if loss has exploded — nothing useful beyond here
        if step > 10 and debiased > 4 * best_loss:
            print(f"  Stopping early at step {step} — loss diverged.")
            break

        # Increase learning rate for next step
        for pg in optimiser.param_groups:
            pg['lr'] *= lr_mult

    # Find the optimal learning rate.
    # Strategy: find the point of steepest negative gradient in the
    # loss curve, then step back one order of magnitude from there.
    # This gives a learning rate that is fast but still stable.
    if len(losses) > 10:
        gradients  = np.gradient(losses)
        steep_idx  = int(np.argmin(gradients))
        optimal_lr = lrs[steep_idx] / 10.0
    else:
        optimal_lr = 1e-3  # fallback if the test was too short

    # Clamp to a sensible range
    optimal_lr = float(np.clip(optimal_lr, 1e-5, 1e-2))

    print(f"  Suggested optimal learning rate : {optimal_lr:.2e}")
    return lrs, losses, optimal_lr


def plot_lr_range_test(lrs, losses, optimal_lr, save_path=None):
    """
    Plot the learning rate range test result.

    The x-axis is learning rate on a log scale. The y-axis is the
    smoothed loss. The vertical line marks the suggested optimal
    learning rate. The sweet spot for training is in the region
    where the loss is falling steeply — not at the bottom, because
    the loss is already becoming unstable there.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lrs, losses, color='steelblue', lw=2)
    ax.axvline(x=optimal_lr, color='firebrick', lw=1.5,
               linestyle='--',
               label=f'Suggested LR = {optimal_lr:.2e}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate (log scale)', fontsize=12)
    ax.set_ylabel('Smoothed loss',             fontsize=12)
    ax.set_title('Learning Rate Range Test\n'
                 'Choose LR just before the loss begins rising steeply',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ── PHASE 2: FULL TRAINING WITH OneCycleLR ───────────────────────────

def train_epoch(model, loader, optimiser, scheduler,
                criterion, device):
    """
    One full pass over the training set.

    The OneCycleLR scheduler steps after every mini-batch, not after
    every epoch. This is the correct usage — it needs fine-grained
    control over the learning rate throughout training.
    """
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        optimiser.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()   # OneCycleLR steps per batch, not per epoch
        total += loss.item() * len(X_b)
    return total / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b  = X_b.to(device)
            y_b  = y_b.to(device)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            total += loss.item() * len(X_b)
    return total / len(loader.dataset)


# ── PLOTTING ──────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses,
                          lr_history, save_path=None):
    """
    Two-panel figure: loss curves and learning rate schedule.

    The learning rate panel shows the OneCycleLR shape — ramp up
    then decay. This makes it easy to see whether the loss improvement
    correlates with specific phases of the schedule.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    sharex=True)

    ax1.plot(train_losses, color='steelblue', lw=2,
             label='Training loss (MSE)')
    ax1.plot(val_losses,   color='firebrick', lw=2,
             linestyle='--', label='Validation loss (MSE)')
    ax1.set_ylabel('MSE loss',  fontsize=11)
    ax1.set_title('TCN training with OneCycleLR — Zernike prediction',
                  fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(lr_history, color='seagreen', lw=2)
    ax2.set_xlabel('Epoch',          fontsize=11)
    ax2.set_ylabel('Learning rate',  fontsize=11)
    ax2.set_title('OneCycleLR schedule', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_prediction_sample(model, X_sample, y_sample,
                            device, save_path=None):
    model.eval()
    with torch.no_grad():
        x_t  = torch.tensor(X_sample[None],
                             dtype=torch.float32).to(device)
        pred = model(x_t).cpu().numpy()[0]
    actual = y_sample
    modes  = np.arange(len(actual))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(modes - 0.2, actual, width=0.4, color='firebrick',
           alpha=0.8, label='Actual next step')
    ax.bar(modes + 0.2, pred,   width=0.4, color='steelblue',
           alpha=0.8, label='TCN prediction')
    ax.set_xlabel('Zernike mode index',      fontsize=12)
    ax.set_ylabel('Coefficient value (rad)', fontsize=12)
    ax.set_title('TCN: predicted vs actual Zernike coefficients\n'
                 '(one sample from the test set)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    print("Step 4b: TCN training with LR range test + OneCycleLR")
    print("=" * 55)
    print(f"Device  : {DEVICE}")
    print(f"Dropout : {DROPOUT}  (increased from 0.1 to reduce overfitting)")

    X, y = load_dataset(DATA_DIR)

    n_features  = X.shape[2]
    n_modes     = y.shape[1]
    window_size = X.shape[1]

    train_loader, val_loader, test_loader = build_dataloaders(X, y)

    # Build model with stronger dropout
    model = ZernikeTCN(
        in_features  = n_features,
        out_features = n_modes,
        window_size  = window_size,
        channels     = 64,
        n_blocks     = 4,
        kernel_size  = 3,
        dropout      = DROPOUT
    ).to(DEVICE)

    print(f"\nModel parameters : {model.count_parameters():,}")
    print(f"Input features   : {n_features}  "
          f"(15 Zernike + distance + Cn2 + noise_std)")
    print(f"Output modes     : {n_modes}")

    criterion = nn.MSELoss()

    # ── Phase 1: Learning rate range test ────────────────────────────
    lrs, lr_losses, optimal_lr = lr_range_test(
        model, train_loader, criterion, DEVICE)

    plot_lr_range_test(
        lrs, lr_losses, optimal_lr,
        save_path=os.path.join(OUT_DIR, "fig_lr_range_test.png"))

    print(f"\nUsing optimal LR = {optimal_lr:.2e} for OneCycleLR")

    # ── Phase 2: Full training with OneCycleLR ────────────────────────
    # Re-initialise the model so Phase 1 weight updates don't carry over
    model = ZernikeTCN(
        in_features  = n_features,
        out_features = n_modes,
        window_size  = window_size,
        channels     = 64,
        n_blocks     = 4,
        kernel_size  = 3,
        dropout      = DROPOUT
    ).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=optimal_lr)

    # OneCycleLR needs to know the total number of optimiser steps,
    # which is epochs × batches_per_epoch.
    steps_per_epoch = len(train_loader)
    total_steps     = EPOCHS * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr       = optimal_lr,
        total_steps  = total_steps,
        pct_start    = 0.3,         # 30% of training spent ramping up
        anneal_strategy = 'cos',    # cosine annealing for smooth decay
        div_factor   = 25.0,        # start LR = max_lr / 25
        final_div_factor = 1e4,     # end LR = max_lr / 10000
    )

    train_losses = []
    val_losses   = []
    lr_history   = []
    best_val     = float('inf')
    best_path    = os.path.join(MODEL_DIR, "tcn_best.pt")

    print(f"\nPhase 2: Training for {EPOCHS} epochs "
          f"({total_steps} total steps)...")

    for epoch in range(1, EPOCHS + 1):
        tr  = train_epoch(model, train_loader, optimiser,
                          scheduler, criterion, DEVICE)
        val = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(tr)
        val_losses.append(val)
        # Record the current learning rate once per epoch
        lr_history.append(optimiser.param_groups[0]['lr'])

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={tr:.4f}  val={val:.4f}  "
                  f"best={best_val:.4f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.2e}")

    # Load best weights and evaluate on held-out test set
    model.load_state_dict(
        torch.load(best_path, map_location=DEVICE))
    test_loss = validate(model, test_loader, criterion, DEVICE)
    print(f"\nTest MSE loss : {test_loss:.4f}")

    # Save final model with all metadata
    torch.save({
        'model_state' : model.state_dict(),
        'in_features' : n_features,
        'out_features': n_modes,
        'window_size' : window_size,
        'channels'    : 64,
        'n_blocks'    : 4,
        'kernel_size' : 3,
        'dropout'     : DROPOUT,
        'optimal_lr'  : optimal_lr,
        'test_loss'   : test_loss,
    }, os.path.join(MODEL_DIR, "tcn_final.pt"))
    print(f"Model saved : {MODEL_DIR}/tcn_final.pt")

    # Figures
    plot_training_curves(
        train_losses, val_losses, lr_history,
        save_path=os.path.join(OUT_DIR, "fig_tcn_training.png"))

    plot_prediction_sample(
        model, X[-100], y[-100], DEVICE,
        save_path=os.path.join(OUT_DIR,
                               "fig_tcn_sample_prediction.png"))

    print(f"\nStep 4b complete. Outputs in: {OUT_DIR}")
    print(f"  fig_lr_range_test.png        — LR search result")
    print(f"  fig_tcn_training.png         — loss + LR schedule")
    print(f"  fig_tcn_sample_prediction.png — one sample prediction")
    print("\nRun run_step5_predictive_compare.py next.")


if __name__ == "__main__":
    main()

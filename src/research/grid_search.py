import itertools
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from src.models.pytorch_lstm import MLSTMFCN

def _finite_row_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Keep only samples whose inputs and targets are all finite."""
    x_ok = torch.isfinite(x).view(x.shape[0], -1).all(dim=1)
    y_ok = torch.isfinite(y).view(y.shape[0], -1).all(dim=1)
    return x_ok & y_ok

def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE or +inf if arrays contain non-finite values or empty after masking."""
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        return float("inf")
    try:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except ValueError:
        return float("inf")

def train_and_evaluate(params, input_tensor, target_tensor, device):
    """
    Train and evaluate MLSTMFCN on a train/validation split.
    Returns (val_loss, rmse); uses +inf when combo is invalid/unstable.
    """
    learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes = params

    # 80/20 split
    train_size = max(1, int(0.8 * len(input_tensor)))
    train_input, val_input = input_tensor[:train_size], input_tensor[train_size:]
    train_target, val_target = target_tensor[:train_size], target_tensor[train_size:]

    # Drop any rows containing NaNs/Infs (defensive)
    mask_train = _finite_row_mask(train_input, train_target)
    train_input, train_target = train_input[mask_train], train_target[mask_train]

    mask_val = _finite_row_mask(val_input, val_target)
    val_input, val_target = val_input[mask_val], val_target[mask_val]

    if len(val_input) == 0 or len(train_input) == 0:
        return float("inf"), float("inf")

    model = MLSTMFCN(
        input_size=int(input_tensor.shape[2]),
        conv_filters=conv_filters,
        kernel_sizes=kernel_sizes,
        lstm_hidden_sizes=lstm_hidden_sizes,
        output_size=int(target_tensor.shape[1])
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 50
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_input.to(device))
        loss = criterion(outputs, train_target.to(device))

        if not torch.isfinite(loss):
            return float("inf"), float("inf")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent explosions
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_input.to(device))
        val_loss = criterion(val_outputs, val_target.to(device))

    val_outputs_np = val_outputs.detach().cpu().numpy()
    val_target_np  = val_target.detach().cpu().numpy()
    rmse = _safe_rmse(val_target_np, val_outputs_np)

    if not np.isfinite(val_loss.item()):
        return float("inf"), float("inf")

    return float(val_loss.item()), rmse

def perform_grid_search(input_tensor, target_tensor, device):
    """
    Manual grid search for MLSTMFCN hyperparameters.
    """
    param_grid = {
        'learning_rate': [0.001, 0.01],
        'conv_filters': [
            [128, 256, 128],
            [64, 128, 64],
        ],
        'kernel_sizes': [
            [90, 60, 30, 20, 8, 5, 3],
            [60, 30, 20, 10, 5, 3],
        ],
        'lstm_hidden_sizes': [
            [128, 128],
            [256, 128],
        ]
    }

    # Filter kernel lists to those valid for the current sequence length
    seq_len = int(input_tensor.shape[1])
    ks_candidates = [ks for ks in param_grid['kernel_sizes'] if max(ks) <= seq_len]
    if not ks_candidates:
        ks_candidates = [[min(seq_len, 30), 20, 10, 5, 3]]

    combos = []
    for lr in param_grid['learning_rate']:
        for cf in param_grid['conv_filters']:
            for ks in ks_candidates:
                for lstm in param_grid['lstm_hidden_sizes']:
                    combos.append((lr, cf, ks, lstm))

    best_params = None
    best_rmse = float('inf')
    results = []

    total = len(combos)
    for i, params in enumerate(combos, start=1):
        (learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes) = params
        print(f"\nTesting combination {i}/{total}: "
              f"LR={learning_rate}, conv_filters={conv_filters}, kernel_sizes={kernel_sizes}, lstm={lstm_hidden_sizes}")

        val_loss, rmse = train_and_evaluate(params, input_tensor, target_tensor, device)
        results.append((params, val_loss, rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

        print(f"Validation Loss: {val_loss:.4f}, RMSE: {rmse:.4f}")

    print(f"\nGrid Search Complete: Best RMSE: {best_rmse:.4f}")
    print(f"Best Params: {best_params}")

    results_df = pd.DataFrame(results, columns=['Params', 'Val Loss', 'RMSE'])
    results_df.to_csv('grid_search_results.csv', index=False)

    return best_params, best_rmse

import itertools
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from src.models.pytorch_lstm import MLSTMFCN

def train_and_evaluate(params, input_tensor, target_tensor, device):
    """
    Train and evaluate MLSTMFCN on a train/validation split.

    :param params: Tuple of hyperparameters (learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes)
    :param input_tensor: preprocessed input features (torch.Tensor)
    :param target_tensor: corresponding targets (torch.Tensor)
    :param device: torch.device
    :return: (val_loss, rmse) as floats
    """
    learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes = params

    # Simple 80/20 split for validation
    train_size = int(0.8 * len(input_tensor))
    train_input, val_input = input_tensor[:train_size], input_tensor[train_size:]
    train_target, val_target = target_tensor[:train_size], target_tensor[train_size:]

    # Define the model with the given hyperparameters
    model = MLSTMFCN(
        input_size=input_tensor.shape[2],
        conv_filters=conv_filters,
        kernel_sizes=kernel_sizes,
        lstm_hidden_sizes=lstm_hidden_sizes,
        output_size=target_tensor.shape[1]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop (can tune epochs here)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_input.to(device))
        loss = criterion(outputs, train_target.to(device))
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_input.to(device))
        val_loss = criterion(val_outputs, val_target.to(device))

    # Compute RMSE
    val_outputs_np = val_outputs.cpu().numpy()
    val_target_np = val_target.cpu().numpy()
    rmse = np.sqrt(mean_squared_error(val_target_np, val_outputs_np))

    return val_loss.item(), rmse

def perform_grid_search(input_tensor, target_tensor, device):
    """
    Perform a manual grid search on MLSTMFCN hyperparameters.
    You can adjust the param_grid to suit your experiments.
    """
    # Example hyperparameter grid
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

    # Create combinations of all parameter values
    param_combinations = list(itertools.product(*param_grid.values()))
    best_params = None
    best_rmse = float('inf')
    results = []

    for i, params in enumerate(param_combinations, start=1):
        (learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes) = params
        print(f"\nTesting combination {i}/{len(param_combinations)}: "
              f"LR={learning_rate}, conv_filters={conv_filters}, kernel_sizes={kernel_sizes}, lstm={lstm_hidden_sizes}")

        val_loss, rmse = train_and_evaluate(params, input_tensor, target_tensor, device)
        results.append((params, val_loss, rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

        print(f"Validation Loss: {val_loss:.4f}, RMSE: {rmse:.4f}")

    print(f"\nGrid Search Complete: Best RMSE: {best_rmse:.4f}")
    print(f"Best Params: {best_params}")

    # Save results to CSV for reference
    results_df = pd.DataFrame(results, columns=['Params', 'Val Loss', 'RMSE'])
    results_df.to_csv('grid_search_results.csv', index=False)

    return best_params, best_rmse
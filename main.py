#%%
import os
import pandas as pd
import numpy as np
import torch
import torch.optim
from torch import nn
from dotenv import load_dotenv  # to load environment variables

# Data loading and processing
from src.data_loader.data_fetcher import YahooFinanceDataLoader, CsvLoader
from src.utils.processing_data import PytorchDataProcessor

# Models and training
from src.models.pytorch_lstm import LSTMModel, MLSTMFCN
from src.trainer.train import PytorchTrainer

# Visualization (optional imports remain here)
from src.analyzer.stock_analyzer import StockPriceAnalyzer
from src.visualizer.draw_price import DrawPrice, DrawTrainingData
from src.visualizer.draw_analysis import DrawMovingAverage

# Import grid search functionality
from src.research.grid_search import perform_grid_search

#%%
# MODE CONFIGURATION
# Set MODE to 'train' for normal training or 'grid' for hyperparameter tuning via grid search.
MODE = 'grid'  # "train" or "grid"

#%%
# Load environment variables from .env file
load_dotenv()

# Basic settings
companies_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
DATA_PATH = os.getenv("DATA_PATH")
if DATA_PATH is None:
    raise ValueError("DATA_PATH is not defined in your .env file!")

TRAINING_WINDOW_SIZE = 90
TARGET_WINDOW_SIZE = 10

# Using CSV file to get data
csv_file_list = [
    f"{DATA_PATH}/AAPL.csv",
    f"{DATA_PATH}/GOOG.csv",
    f"{DATA_PATH}/MSFT.csv",
    f"{DATA_PATH}/AMZN.csv"
]
csv_data_loader = CsvLoader(csv_file_list)
data_aapl = csv_data_loader.get_data(company_name='AAPL', start_date='2020-01-01', end_date='2023-05-31')

# Process data
data_processor = PytorchDataProcessor(
    input_data=data_aapl,
    extract_column=['Close', 'Volume'],  # multiple columns
    training_data_ratio=0.8,
    training_window_size=TRAINING_WINDOW_SIZE,
    target_window_size=TARGET_WINDOW_SIZE
)
data_processor.process_data()
input_tensor, target_tensor = data_processor.get_tensors_to_train_model()

print("Input tensor shape:", input_tensor.shape)
print("Target tensor shape:", target_tensor.shape)

# Determine device (default to 'mps' as in original code)
device = torch.device('mps')
print(f"Using device: {device}")

#%%
# Decide between normal training or grid search
EPOCHS_FINAL = 300  # We will train for 300 epochs in both modes

if MODE == 'grid':
    print("Running Grid Search Hyperparameter Tuning on MLSTMFCN...")
    best_params, best_rmse = perform_grid_search(input_tensor, target_tensor, device=device)
    print(f"\nBest Parameters Found by Grid Search: {best_params}")
    print(f"Best RMSE on Validation Set: {best_rmse:.4f}")

    # Unpack the best hyperparameters.
    # Make sure the order matches your param_grid in grid_search.py.
    learning_rate, conv_filters, kernel_sizes, lstm_hidden_sizes = best_params

    # Re-train a final model with the best hyperparameters for EPOCHS_FINAL
    print(f"\nRe-training final MLSTMFCN model with the best hyperparameters for {EPOCHS_FINAL} epochs...")
    final_model = MLSTMFCN(
        input_size=input_tensor.shape[2],
        conv_filters=conv_filters,
        kernel_sizes=kernel_sizes,
        lstm_hidden_sizes=lstm_hidden_sizes,
        output_size=target_tensor.shape[1]  # If your target has shape (N, 10)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    final_trainer = PytorchTrainer(
        model=final_model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        training_data=input_tensor,
        training_labels=target_tensor
    )
    final_trainer.run_training_loop(epochs=EPOCHS_FINAL)
    final_model.eval()

else:
    # NORMAL TRAINING
    print(f"Running Normal Training Pipeline for {EPOCHS_FINAL} epochs using fixed hyperparameters...")

    # Default hyperparameters
    conv_filters = [128, 256, 128]
    kernel_sizes = [90, 60, 30, 20, 8, 5, 3]
    lstm_hidden_sizes = [128, 128]
    learning_rate = 0.001

    final_model = MLSTMFCN(
        input_size=input_tensor.shape[2],
        conv_filters=conv_filters,
        kernel_sizes=kernel_sizes,
        lstm_hidden_sizes=lstm_hidden_sizes,
        output_size=10  # Predicting next 10 days' price
    ).to(device)

    final_model.eval()

    trainer = PytorchTrainer(
        model=final_model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam(final_model.parameters(), lr=learning_rate),
        device=device,
        training_data=input_tensor,
        training_labels=target_tensor
    )

    trainer.run_training_loop(epochs=EPOCHS_FINAL)
    final_model.eval()

#%%
# Continue with final model testing and plotting/animation (original style preserved)
print("\nEvaluating final model on test set...")

test_tensor, test_target = data_processor.get_tensors_to_test_model()
test_tensor = test_tensor.to(device)
prediction = final_model(test_tensor).to('cpu').detach().numpy()

# **Important**: If the target dimension includes multiple columns (e.g., 'Close' and 'Volume'),
# we must select only the "Close" portion of the target for plotting.
if test_target.ndim == 3 and test_target.shape[-1] > 1:
    test_target = test_target[..., 0]

# Inverse scaling for plotting
test_tensor_inv = data_processor.inverse_testing_scaler(
    data=test_tensor.to('cpu')[:, :, 0],  # Already picking the [close] dimension
    scaler_by_column_name='Close'
)
prediction_output = data_processor.inverse_testing_scaler(
    data=prediction,
    scaler_by_column_name='Close'
)
test_target_inv = data_processor.inverse_testing_scaler(
    data=test_target.numpy(),
    scaler_by_column_name='Close'
)

from matplotlib import pyplot as plt
plt.plot(test_target_inv[:, 9].reshape(-1, 1), label="ground true")
plt.plot(prediction_output[:, 9].reshape(-1, 1), label="prdiction")
plt.legend()
plt.show()

#%%
training_set_df = data_processor.get_training_set()
testing_set_df = data_processor.get_testing_set()

training_set_df.reset_index(inplace=True)
testing_set_df.reset_index(inplace=True)

print(training_set_df)
print(testing_set_df)

training_dates_list = training_set_df['Date'].tolist()
close_price_series_training_list = training_set_df['Close'].tolist()
testing_dates_list = testing_set_df['Date'].tolist()
close_price_series_test_list = testing_set_df['Close'].tolist()

#%%
# Future prediction visualization (static plots) in original style
from datetime import datetime, timedelta
testing_start_date = testing_dates_list[0]

for i in range(test_tensor_inv.shape[0] - 100):
    plt.figure(figsize=(16, 9))
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.grid()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("AAPL Stock Prices: Training and Testing Segments")

    historical_x_list = [testing_start_date + timedelta(days=x) for x in range(90)]
    future_x_list = [historical_x_list[-1] + timedelta(days=x) for x in range(11)]
    prediction_day = historical_x_list[-1]

    historical_data = test_tensor_inv[i, :].tolist()      # shape (90,) of floats
    future_data_target = test_target_inv[i, :].tolist()   # shape (10,) of floats
    future_prediction = prediction_output[i, :].tolist()  # shape (10,) of floats

    # Plot historical data
    plt.plot(historical_x_list, historical_data, label='Historical Data', color='blue')
    future_data_target.insert(0, historical_data[-1])
    future_prediction.insert(0, historical_data[-1])
    # Plot testing data and predictions
    plt.plot(future_x_list, future_data_target, label='Testing Data', color='red')
    plt.plot(future_x_list, future_prediction, label='LSTM model Prediction', color='orange')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./output_fig/stock_price_predict_{prediction_day.strftime('%Y_%m_%d')}.png")
    
    print(testing_start_date)
    testing_start_date = testing_start_date + timedelta(days=1)
    print(testing_start_date)

#%%
# Animation section (original style preserved)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import timedelta
import os

fig, ax = plt.subplots(figsize=(16, 9))
ax.grid()
ax.set_title("AAPL Stock Price Prediction Animation")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")

historical_line, = ax.plot([], [], label='Historical Data', color='blue')
true_line, = ax.plot([], [], label='True Future Data', color='red')
prediction_line, = ax.plot([], [], label='Predicted Data', color='orange')
ax.legend()

def update(frame):
    if frame >= len(test_tensor_inv):
        return
    historical_x = [testing_start_date - timedelta(days=90) + timedelta(days=frame + i) for i in range(90)]
    future_x = [testing_start_date + timedelta(days=frame + i) for i in range(10)]
    historical_data = test_tensor_inv[frame, :].tolist()
    true_future_data = test_target_inv[frame, :].tolist()
    predicted_future_data = prediction_output[frame, :].tolist()

    true_future_data.insert(0, historical_data[-1])
    predicted_future_data.insert(0, historical_data[-1])
    future_x.insert(0, historical_x[-1])

    historical_line.set_data(historical_x, historical_data)
    true_line.set_data(future_x, true_future_data)
    prediction_line.set_data(future_x, predicted_future_data)
    
    ax.set_xlim(historical_x[0], future_x[-1])
    all_data = historical_data + true_future_data + predicted_future_data
    ax.set_ylim(min(all_data) * 0.95, max(all_data) * 1.05)

output_dir = "./output_animation"
os.makedirs(output_dir, exist_ok=True)
ani = FuncAnimation(fig, update, frames=len(test_tensor_inv)-100, interval=200, repeat=False)
ani.save(f"{output_dir}/stock_price_prediction_animation.mp4", writer="ffmpeg", fps=10)
plt.show()
#%%
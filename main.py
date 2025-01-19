#%%
import os
import pandas as pd
import numpy as np
import torch.optim
from torch import nn
from dotenv import load_dotenv  # to load environment variables

from src.data_loader.data_fetcher import YahooFinanceDataLoader, CsvLoader
from src.analyzer.stock_analyzer import StockPriceAnalyzer
from src.visualizer.draw_price import DrawPrice, DrawTrainingData
from src.visualizer.draw_analysis import DrawMovingAverage

from src.models.pytorch_lstm import LSTMModel, MLSTMFCN
from src.trainer.train import PytorchTrainer
from src.utils.processing_data import PytorchDataProcessor
#%%
# Load environment variables from .env file
load_dotenv()

companies_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# using yahoo finance to get data

DATA_PATH = os.getenv("DATA_PATH")
if DATA_PATH is None:
    raise ValueError("DATA_PATH is not defined in your .env file!")

TRAINING_WINDOW_SIZE = 90
TARGET_WINDOW_SIZE = 10

# yf_data_loader = YahooFinanceDataLoader(start_date='2000-01-01', end_date='2023-06-01')
# yf_data_loader.get_compamy_data_and_save_to_csv(company_list=companies_list, path_to_save_data=DATA_PATH)

# list_of_data = yf_data_loader.get_company_data(company_list=companies_list)

# using csv file to get data
csv_file_list = [
    f"{DATA_PATH}/AAPL.csv",
    f"{DATA_PATH}/GOOG.csv",
    f"{DATA_PATH}/MSFT.csv",
    f"{DATA_PATH}/AMZN.csv"
]
csv_data_loader = CsvLoader(csv_file_path_list=csv_file_list)
data_aapl = csv_data_loader.get_data(company_name='AAPL', start_date='2020-01-01', end_date='2023-05-31')

data_processor = PytorchDataProcessor(
    input_data=data_aapl,
    extract_column=['Close', 'Volume'],
    training_data_ratio=0.8,
    training_window_size=TRAINING_WINDOW_SIZE,
    target_window_size=TARGET_WINDOW_SIZE
)

data_processor.process_data()
input_tensor, target_tensor = data_processor.get_tensors_to_train_model()

print(input_tensor.shape)
print(target_tensor.shape)

#%%

#%%
# define the model

# model = LSTMModel(
#     input_size = input_tensor.shape[2],
#     hidden_layer_size=[128, 256, 128],
#     output_size=10
# )

# %%
# Define the MLSTMFCN model
# Note: You need to define the conv_filters and lstm_hidden_sizes according to your model's architecture
conv_filters = [128, 256, 128]  # Number of filters for each conv layer
kernel_sizes = [90, 60, 30, 20, 8, 5, 3]  # Kernel sizes for each conv layerfilters and kernel sizes
lstm_hidden_sizes = [128, 128]  # Example: 2 LSTM layers with specified hidden sizes

model = MLSTMFCN(
    input_size=input_tensor.shape[2],
    conv_filters=conv_filters,
    kernel_sizes=kernel_sizes,
    lstm_hidden_sizes=lstm_hidden_sizes,
    output_size=10  # Predicting the stock price for the next 10 days
)

# %%
model.eval()

# %%
trainer = PytorchTrainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device=torch.device('mps'),
    training_data=input_tensor,
    training_labels=target_tensor
)

# %%

trainer.run_training_loop(epochs=300)

model.eval()

#%%

test_tensor, test_target = data_processor.get_tensors_to_test_model()

test_tensor = test_tensor.to(torch.device('mps'))
prediction = model(test_tensor).to('cpu').detach().numpy()


#%%
test_tensor = data_processor.inverse_testing_scaler(
    data=test_tensor.to('cpu')[:,:,0],
    scaler_by_column_name='Close'
)

prediction_output = data_processor.inverse_testing_scaler(
    data=prediction,
    scaler_by_column_name='Close'
)
test_target = data_processor.inverse_testing_scaler(
    data=test_target.numpy(),
    scaler_by_column_name='Close'
)

from matplotlib import pyplot as plt

plt.plot(test_target[:, 9].reshape(-1, 1), label="ground true")
plt.plot(prediction_output[:, 9].reshape(-1, 1), label="prdiction")
plt.show()


# %%

training_set_df = data_processor.get_training_set()
testing_set_df = data_processor.get_testing_set()

training_set_df.reset_index(inplace=True)
testing_set_df.reset_index(inplace=True)

#%%
training_set_df
#%%
testing_set_df

#%%

training_dates_list = training_set_df['Date'].tolist()
close_price_series_training_list = training_set_df['Close'].tolist()
testing_dates_list = testing_set_df['Date'].tolist()
close_price_series_test_list = testing_set_df['Close'].tolist()

#%%
# future_prediction

#%%
from datetime import datetime, timedelta

testing_start_date = testing_dates_list[0]

#%%

for i in range(test_tensor.shape[0]-100):
    plt.figure(figsize=(16, 9))
    plt.gcf().autofmt_xdate()  # Auto rotate date labels
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Formatting date display
    plt.grid()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("AAPL Stock Prices: Training and Testing Segments")


    historical_x_list = [testing_start_date + timedelta(days=x) for x in range(90)] # Extend 90 days from testing_start_date
    future_x_list = [historical_x_list[-1] + timedelta(days=x) for x in range(11) ]

    prediction_day = historical_x_list[-1]
    

    historical_data = test_tensor[i,:].tolist()
    future_data_target = test_target[i,:].tolist()
    future_prediction = prediction_output[i,:].tolist()

    # Plot training data
    plt.plot(historical_x_list, historical_data, label='Historical Data', color='blue')

    future_data_target.insert(0, historical_data[-1])
    future_prediction.insert(0, historical_data[-1])
    # close_price_series_test_list.insert(0, close_price_series_training_list[-1])
    # Plot testing data
    plt.plot(future_x_list, future_data_target, label='Testing Data', color='red')
    plt.plot(future_x_list, future_prediction, label='LSTM model Prediction', color='orange')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./output_fig/stock_price_predict_{str(prediction_day.strftime('%Y_%m_%d'))}_.png")

    print(testing_start_date)
    testing_start_date = testing_start_date + timedelta(days=1)
    print(testing_start_date)


    # # Formatting date labels
    # plt.gcf().autofmt_xdate()  # Auto rotate date labels
    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Formatting date display

    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.title("AAPL Stock Prices: Training and Testing Segments")
    # plt.legend()
    # plt.show()
    # plt.close()

    # # Move 10 elements from testing to training, if available
    # move_count = min(10, len(testing_dates_list))
    # training_dates_list.extend(testing_dates_list[:move_count])
    # close_price_series_training_list.extend(close_price_series_test_list[:move_count])

    # # Remove the moved elements from the testing lists
    # testing_dates_list = testing_dates_list[move_count:]
    # close_price_series_test_list = close_price_series_test_list[move_count:]
    # break
# %%

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
    if frame >= len(test_tensor):
        return
    historical_x = [testing_start_date - timedelta(days=90) + timedelta(days=frame + i) for i in range(90)]
    future_x = [testing_start_date + timedelta(days=frame + i) for i in range(10)]
    historical_data = test_tensor[frame, :].tolist()
    true_future_data = test_target[frame, :].tolist()
    predicted_future_data = prediction_output[frame, :].tolist()

    true_future_data = [historical_data[-1]] + true_future_data
    predicted_future_data = [historical_data[-1]] + predicted_future_data
    future_x = [historical_x[-1]] + future_x

    historical_line.set_data(historical_x, historical_data)
    true_line.set_data(future_x, true_future_data)
    prediction_line.set_data(future_x, predicted_future_data)
    
    ax.set_xlim(historical_x[0], future_x[-1])
    all_data = historical_data + true_future_data + predicted_future_data
    ax.set_ylim(min(all_data) * 0.95, max(all_data) * 1.05)

output_dir = "./output_animation"
os.makedirs(output_dir, exist_ok=True)

ani = FuncAnimation(fig, update, frames=len(test_tensor)-100, interval=200, repeat=False)
ani.save(f"{output_dir}/stock_price_prediction_animation.mp4", writer="ffmpeg", fps=10)
plt.show()
# %%

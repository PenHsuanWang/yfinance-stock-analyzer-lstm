# yfinance-stock-analyzer-lstm

This guide on using deep learning to predict stock prices! In this project, you will learn how to use LSTM-based models to perform time series forecasting of stock prices. This guide will take you step-by-step through the data loading, preprocessing, model training, and visualization of results.

## Overview

This project includes the following key components:

1. **Data Loading**: We start by loading stock data from Yahoo Finance or CSV files.

   - The `YahooFinanceDataLoader` fetches data directly from Yahoo Finance.
   - Alternatively, the `CsvLoader` can load saved CSV files containing historical stock data.

2. **Data Preprocessing**: We preprocess the data to make it suitable for training our deep learning models.

   - The `PytorchDataProcessor` is used to select the columns we want (`Close` and `Volume`), scale the data, and split it into training and testing sets using sliding windows.

3. **Model Definition**: We provide two LSTM-based models to predict stock prices based on historical data.

   - **LSTMModel**: A simple yet powerful LSTM model that consists of multiple LSTM layers and a fully connected layer to generate predictions. This model captures the temporal dependencies of time-series data.
   - **MLSTMFCN**: A more sophisticated model combining LSTM layers and convolutional branches to extract both temporal and spatial features. This hybrid model is well-suited for complex time-series forecasting tasks. It includes:
     - **Convolutional Branch**: Uses `Conv1DBlock` layers for feature extraction, followed by `SqueezeExciteBlock` layers to enhance the most important features.
     - **LSTM Branch**: Includes LSTM layers to capture temporal patterns in the stock data.
     - **Concatenation and Output**: Combines the convolutional and LSTM features to generate the final predictions.

4. **Training**: You will train the models using the `PytorchTrainer` class, which handles the training loop, optimizer setup, and loss function.

5. **Prediction and Visualization**: After training, we will use the model to predict future stock prices and visualize the results using `matplotlib`.

## Getting Started

### Prerequisites

Before we start, make sure you have the following installed:

- Python 3.x
- PyTorch
- Pandas, NumPy
- Matplotlib
- Yahoo Finance API (optional)

To install the required libraries, run:

### Project Structure

- **src/data_loader**: Contains the classes for loading stock data (`YahooFinanceDataLoader`, `CsvLoader`).
- **src/analyzer**: Provides tools for analyzing stock data.
- **src/visualizer**: Includes modules for visualizing stock data (`DrawPrice`, `DrawTrainingData`, etc.).
- **src/models**: Contains the model definitions (`LSTMModel`, `MLSTMFCN`).
- **src/trainer**: Includes `PytorchTrainer` to train the models.
- **src/utils**: Utility functions for data processing (`PytorchDataProcessor`).

### Step-by-Step Guide

#### Step 1: Data Preparation

1. **Specify the Companies**: Start by defining the list of companies whose stock data you want to analyze.

2. **Fetch and Save Stock Data**: Use the YahooFinanceDataLoader to download data and save it locally.

3. **Load Data from CSV**: Load the saved data using the CsvLoader.

#### Step 2: Data Preprocessing

- **Preprocess the Data**: Use `PytorchDataProcessor` to process the data for training.

  This step will normalize the data and create training and testing sets based on sliding windows.

#### Step 3: Model Training

- **Define the Model**: You can choose between `LSTMModel` and `MLSTMFCN`. Here, we use `MLSTMFCN` for demonstration.

- **Set Up the Trainer**: Initialize the `PytorchTrainer` to handle the training process.

- **Train the Model**: Run the training loop for the desired number of epochs.

#### Step 4: Prediction and Visualization

- **Make Predictions**: Using the model to predict future stock prices after training.

- **Visualize the Results**: Use `matplotlib` to visualize the actual and predicted stock prices.

### Understanding the Visualization

The visualizations generated will show the actual stock prices compared to the predicted values from the model. This helps in evaluating how well the model has learned the underlying patterns in the stock data.

#### Detailed Plotting Instructions

When visualizing stock prices, it's essential to clearly separate different phases such as historical data, testing data, and predictions. In the example provided, we use different colors and labels to distinguish these segments.

- **Historical Data**: This represents the data that was used during training. It is plotted in blue to provide context on the model's training inputs.
- **Testing Data**: The actual stock prices during the testing period are plotted in red. This allows you to directly compare the predictions against the ground truth.
- **Predicted Data**: The model's predictions for the testing period are plotted in orange. This visualization provides insights into how well the model can generalize and forecast future trends.

Below is a detailed example that includes advanced features such as plotting date labels, adding grids, and saving the plots to files:

This example uses the following features:

- **Auto Rotate Date Labels**: `plt.gcf().autofmt_xdate()` ensures the date labels are readable.
- **Grid**: `plt.grid()` makes it easier to see the variations in stock prices.
- **Saving the Plot**: `plt.savefig()` Saves each plot as a PNG file, which is useful for keeping a record of predictions over time.

## Advanced Topics

### Improving the Model

- **Try Different Architectures: Consider using more advanced architectures like Transformers to improve prediction accuracy.**
- **Hyperparameter Tuning: Experiment with different hyperparameters, such as learning rate, batch size, and the number of LSTM layers, to find the best configuration for your model.**
- **Predict Multiple Stocks**: By modifying the output layer, you can extend the model to predict multiple stocks simultaneously.


## Getting Started

The sample code is provided in the `main.py` file. The sample code is to analyze the stock price of Apple Inc. (AAPL) from 2010-01-01 to 2019-12-31. The sample code is to predict the stock price of AAPL in 2020-01-01 to 2020-01-31.

`main.py`
```python

import pandas as pd
import numpy as np
import torch.optim
from torch import nn

from src.data_loader.data_fetcher import YahooFinanceDataLoader, CsvLoader
from src.analyzer.stock_analyzer import StockPriceAnalyzer
from src.visualizer.draw_price import DrawPrice, DrawTrainingData
from src.visualizer.draw_analysis import DrawMovingAverage

from src.models.pytorch_lstm import LSTMModel, MLSTMFCN
from src.trainer.train import PytorchTrainer
from src.utils.processing_data import PytorchDataProcessor


# Defining the path to access the csv data

companies_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

DATA_PATH = '<The home path to access csv data>'

TRAINING_WINDOW_SIZE = 90
TARGET_WINDOW_SIZE = 10

csv_file_list = [
    f"{DATA_PATH}/<your_data_download_with_yfinance>.csv"
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


# the input_tensor is the input data for the model
# the target_tensor is the target data for the model

# define the model

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

model.eval()

trainer = PytorchTrainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device=torch.device('mps'),
    training_data=input_tensor,
    training_labels=target_tensor
)

trainer.run_training_loop(epochs=300)

model.eval()

test_tensor, test_target = data_processor.get_tensors_to_test_model()

test_tensor = test_tensor.to(torch.device('mps'))
prediction = model(test_tensor).to('cpu').detach().numpy()

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


training_set_df = data_processor.get_training_set()
testing_set_df = data_processor.get_testing_set()

training_set_df.reset_index(inplace=True)
testing_set_df.reset_index(inplace=True)

training_dates_list = training_set_df['Date'].tolist()
close_price_series_training_list = training_set_df['Close'].tolist()
testing_dates_list = testing_set_df['Date'].tolist()
close_price_series_test_list = testing_set_df['Close'].tolist()

from datetime import datetime, timedelta

testing_start_date = testing_dates_list[0]


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


```


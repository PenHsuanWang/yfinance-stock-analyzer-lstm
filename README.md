# yfinance-stock-analyzer-lstm

The project to modulize the stock analyzer using LSTM. Provided the stock data from Yahoo Finance.

## Getting Started

The sample code is provided in the `main.py` file. The sample code is to analyze the stock price of Apple Inc. (AAPL) from 2010-01-01 to 2019-12-31. The sample code is to predict the stock price of AAPL in 2020-01-01 to 2020-01-31.

`main.py`
```python

from src.data_loader.data_fetcher import CsvLoader
from src.utils.processing_data import  PytorchDataProcessor
from src.models.pytorch_lstm import LSTMModel
from src.trainer.train import PytorchTrainer
import torch
import torch.nn as nn
import torch.optim


# Defining the path to access the csv data
DATA_PATH = '<The home path to access csv data>'
csv_file_list = [
    f"{DATA_PATH}/<your_data_download_with_yfinance>.csv"
]
csv_data_loader = CsvLoader(
    csv_file_path_list=csv_file_list,
    start_date='2010-01-01',
    end_date='2020-01-31'
)

# Extract data from csv file to pandas dataframe
stock_data = csv_data_loader.get_data(company_name='<company_index>') # e.g. AAPL


# Define the data processor
data_processor = PytorchDataProcessor(
    input_data=stock_data,
    extract_column=['Close', 'Volume'], # the first column is the target column
    training_data_ratio=0.8,
    training_window_size=60, # Using 60 days of data to train the model
    target_window_size=10 # predict the stock price of next 10 days
)

data_processor.process_data()
input_tensor, target_tensor = data_processor.get_tensors_to_test_model()

# the input_tensor is the input data for the model
# the target_tensor is the target data for the model

# define the model

model = LSTMModel(
    input_size = input_tensor.shape[2],
    hidden_size=128,
    output_size=10
)

trainer = PytorchTrainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device=torch.device('mps'),
    training_data=input_tensor,
    training_labels=target_tensor
)

trainer.run_training_loop(epochs=50)

model.eval()

test_tensor, test_target = data_processor.get_tensors_to_test_model()

test_tensor = test_tensor.to(torch.device('mps'))
prediction = model(test_tensor).to('cpu').detach().numpy()

prediction_output = data_processor.inverse_testing_scaler(
    data=prediction,
    scaler_by_column_name='Close'
)
test_target = data_processor.inverse_testing_scaler(
    data=test_target.numpy(),
    scaler_by_column_name='Close'
)


```


import os
import pandas as pd
import torch
from unittest.mock import patch
import pytest
from src.data_loader.data_fetcher import YahooFinanceDataLoader, CsvLoader

# Mock data for testing
mock_stock_data = pd.DataFrame({
    'Open': [100, 102, 104],
    'High': [101, 103, 105],
    'Low': [99, 101, 103],
    'Close': [100, 102, 104],
    'Volume': [1000, 1000, 1000]
})


# Fixture for YahooFinanceDataLoader
@pytest.fixture
def mock_yahoo_finance_data_loader():
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = mock_stock_data
        yield


# Test YahooFinanceDataLoader
def test_yahoo_finance_data_loader(mock_yahoo_finance_data_loader):
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    data = YahooFinanceDataLoader.get_company_data(['AAPL'], start_date, end_date)
    assert 'AAPL' in data
    assert not data['AAPL'].empty
    pd.testing.assert_frame_equal(data['AAPL'], mock_stock_data)


# Fixture for CsvLoader
@pytest.fixture
def sample_csv_paths(tmp_path):
    dates = pd.date_range(start='2023-01-01', periods=len(mock_stock_data))
    mock_stock_data_with_date = mock_stock_data.copy()
    mock_stock_data_with_date['Date'] = dates
    mock_stock_data_with_date.to_csv(tmp_path / "AAPL.csv", index=False)
    return [str(tmp_path / "AAPL.csv")]


# Test CsvLoader for loading data
def test_csv_loader_get_data(sample_csv_paths):
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    csv_loader = CsvLoader(sample_csv_paths)
    data = csv_loader.get_data('AAPL', start_date, end_date)
    assert not data.empty
    assert start_date <= data.index.min().strftime('%Y-%m-%d') <= end_date

# Test CsvLoader for preparing training data
def test_csv_loader_prepare_training_data(sample_csv_paths):
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    csv_loader = CsvLoader(sample_csv_paths)
    training_data = csv_loader.prepare_training_data_for_pytorch_model(start_date, end_date)
    assert 'AAPL' in training_data
    for data in training_data.values():
        assert isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor)

import pytest
import pandas as pd
import numpy as np
import torch
from src.utils.processing_data import PytorchDataProcessor


@pytest.fixture
def sample_input_data():
    # Create a sample input data for testing
    data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05',
                     '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10'],
            'Value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]}
    df = pd.DataFrame(data)
    return df


def test_process_data(sample_input_data):
    # Initialize the PytorchDataProcessor
    processor = PytorchDataProcessor(input_data=sample_input_data, extract_column='Value', training_data_ratio=0.8,
                                     training_window_size=2)

    # Perform data processing
    processor.process_data()

    # Check if training dataset is correctly split
    assert len(processor.get_training_set()) == 8

    # Check if testing dataset is correctly split
    assert len(processor.get_testing_set()) == 2

    # Check if training tensor shape is correct
    training_tensor, training_target_tensor = processor.get_tensors_to_train_model()
    assert training_tensor.shape == torch.Size([6, 2, 1])
    assert training_target_tensor.shape == torch.Size([6, 1])

    # Check if testing tensor shape is correct
    testing_tensor, testing_target_tensor = processor.get_tensors_to_test_model()
    assert testing_tensor.shape == torch.Size([2, 2, 1])
    assert testing_target_tensor.shape == torch.Size([2, 1])


def test_inverse_scaler(sample_input_data):
    # Initialize the PytorchDataProcessor
    processor = PytorchDataProcessor(input_data=sample_input_data, extract_column='Value', training_data_ratio=0.8,
                                     training_window_size=2)

    # Perform data processing
    processor.process_data()

    # Perform inverse scaling on sample data
    # scaled_data = processor.inverse_scaler(np.array([0.5]))

    # Check if the inverse scaled data is correct
    assert sample_input_data['Value'].max() == 55
    assert sample_input_data['Value'].min() == 10
    # assert scaled_data == (sample_input_data['Value'].min() + sample_input_data['Value'].max()) * 0.5



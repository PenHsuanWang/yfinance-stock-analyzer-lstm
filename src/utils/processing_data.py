
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch


class BaseDataProcessor(ABC):

    def __init__(self, input_data):
        self._input_data = input_data

    def _scaling_by_column(self):
        raise NotImplementedError

    def _splitting(self):
        raise NotImplementedError

    def _preprocessing(self):
        raise NotImplementedError


class PytorchDataProcessor(BaseDataProcessor):

    def __init__(self, input_data, target_column: str, training_data_ratio: float = 0.8, window_size: int = 60):
        super().__init__(input_data)
        self._target_column = target_column
        self._target_series = None
        self._training_data_ratio = training_data_ratio
        self._window_size = window_size

        self._training_dataset = None
        self._testing_dataset = None

        self._training_tensor = None
        self._training_target_tensor = None

        self._testing_tensor = None
        self._testing_target_tensor = None

    def _extract_training_data(self) -> None:
        self._target_series = self._input_data[self._target_column].values
        if self._target_series is None:
            raise ValueError(f"Please check the column {self._target_column} exist in the input data!")

    def _scaling_by_column(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self._target_series = scaler.fit_transform(self._target_series.reshape(-1, 1))

    def _splitting(self):
        self._training_dataset = self._target_series[:int(self._training_data_ratio * len(self._target_series))]
        self._testing_dataset = self._target_series[int(self._training_data_ratio * len(self._target_series)):]

    def _preprocessing(self):
        x_train, y_train = self._tumbling_window_mask_on_training_data()
        x_test = self._testing_dataset
        y_test = self._testing_dataset[1:]

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self._training_tensor = torch.from_numpy(x_train).float()
        self._training_target_tensor = torch.from_numpy(y_train).float()

        self._testing_tensor = torch.from_numpy(x_test).float()
        self._testing_target_tensor = torch.from_numpy(y_test).float()

    def process_data(self) -> None:
        self._extract_training_data()
        self._scaling_by_column()
        self._splitting()
        self._preprocessing()

    def get_tensors_to_train_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._training_tensor, self._training_target_tensor

    def get_tensors_to_test_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._testing_tensor, self._testing_target_tensor

    def _tumbling_window_mask_on_training_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        x_train = []
        y_train = []

        for i in range(self._window_size, len(self._training_dataset)):
            x_train.append(self._target_series[i - self._window_size:i])
            y_train.append(self._target_series[i])

        return x_train, y_train

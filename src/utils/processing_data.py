
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

        self._training_series = None
        self._testing_series = None

        self._training_dataset = None
        self._testing_dataset = None

        self._training_tensor = None
        self._training_target_tensor = None

        self._testing_tensor = None
        self._testing_target_tensor = None

        self._scaler_training = MinMaxScaler()
        self._scaler_testing = MinMaxScaler()

    def _extract_training_data(self) -> None:
        self._target_series = self._input_data[self._target_column].values
        if self._target_series is None:
            raise ValueError(f"Please check the column {self._target_column} exist in the input data!")

    def _scaling_by_column(self):
        # self._target_series = self._scaler.fit_transform(self._target_series.reshape(-1, 1))
        self._scaler_training.fit(self._training_series.reshape(-1, 1))
        self._scaler_testing.fit_transform(self._testing_series.reshape(-1, 1))
        self._training_series = self._scaler_training.transform(self._training_series.reshape(-1, 1))
        self._testing_series = self._scaler_testing.transform(self._testing_series.reshape(-1, 1))

    def _splitting(self):
        self._training_series = self._target_series[:int(self._training_data_ratio * len(self._target_series))]
        self._testing_series = self._target_series[int(self._training_data_ratio * len(self._target_series))-self._window_size:]

        self._training_dataset = self._input_data[:int(self._training_data_ratio * len(self._target_series))]
        self._testing_dataset = self._input_data[int(self._training_data_ratio * len(self._target_series)):]

    def _preprocessing(self):
        x_train, y_train = self._sliding_window_mask_on_training_data()
        x_test, y_test = self._sliding_window_mask_on_testing_data()

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        self._training_tensor = torch.from_numpy(x_train).float()
        self._training_target_tensor = torch.from_numpy(y_train).float()

        self._testing_tensor = torch.from_numpy(x_test).float()
        self._testing_target_tensor = torch.from_numpy(y_test).float()

    def process_data(self) -> None:
        self._extract_training_data()
        # self._scaling_by_column()
        self._splitting()
        self._scaling_by_column()
        self._preprocessing()

    def get_training_set(self) -> pd.DataFrame:
        return self._training_dataset

    def get_testing_set(self) -> pd.DataFrame:
        return self._testing_dataset

    def get_tensors_to_train_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._training_tensor, self._training_target_tensor

    def get_tensors_to_test_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._testing_tensor, self._testing_target_tensor

    def inverse_testing_scaler(self, data: np.ndarray) -> np.ndarray:
        return self._scaler_testing.inverse_transform(data.reshape(-1, 1))

    def _sliding_window_mask_on_training_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        x_train = []
        y_train = []

        for i in range(self._window_size, len(self._training_series)):
            x_train.append(self._training_series[i - self._window_size:i])
            y_train.append(self._training_series[i])

        return x_train, y_train

    def _sliding_window_mask_on_testing_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        x_test = []
        y_test = []

        for i in range(self._window_size, len(self._testing_series)):
            x_test.append(self._testing_series[i - self._window_size:i])
            y_test.append(self._testing_series[i])

        return x_test, y_test

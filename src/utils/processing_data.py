
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch


class BaseDataProcessor(ABC):

    def __init__(self, input_data):
        self._input_df = input_data

    def _scaling_array(self):
        raise NotImplementedError

    def _splitting(self):
        raise NotImplementedError

    def _preprocessing(self):
        raise NotImplementedError


class PytorchDataProcessor(BaseDataProcessor):

    def __init__(self,
                 input_data,
                 extract_column: list[str],
                 training_data_ratio: float = 0.8,
                 training_window_size: int = 60,
                 target_window_size: int = 1
                 ):
        super().__init__(input_data)

        self._extract_column = extract_column
        self._extract_data = None
        self._training_data_ratio = training_data_ratio
        self._training_window_size = training_window_size
        self._target_window_size = target_window_size

        # intermediate dataset
        self._training_array = None
        self._testing_array = None

        self._training_set_df = None
        self._testing_set_df = None

        self._training_tensor = None
        self._training_target_tensor = None

        self._testing_tensor = None
        self._testing_target_tensor = None

        self._scaler_training = MinMaxScaler()
        self._scaler_testing = MinMaxScaler()

        self._scaler_by_column = {}

    def _extract_training_data_and_scale(self) -> None:

        # if len(self._extract_column) == 1:
        #     self._extract_data = self._input_df[self._extract_column].values
        #     if self._extract_data is None:
        #         raise ValueError(f"Please check the column {self._extract_column} exist in the input data!")
        # else:
        if self._extract_column is None:
            raise ValueError(f"Please check the column {self._extract_column} exist in the input data!")

        for i_column in self._extract_column:
            extract_column_series = self._input_df[i_column].values
            extract_column_series_after_scaling, scaler = self.scaling_one_d_array(extract_column_series)
            extract_column_series_after_scaling = np.expand_dims(extract_column_series_after_scaling.reshape(-1), axis=1)
            self._scaler_by_column[i_column] = scaler
            if self._extract_data is None:
                self._extract_data = extract_column_series_after_scaling
            else:
                self._extract_data = np.concatenate((self._extract_data, extract_column_series_after_scaling), axis=1)

    def _splitting(self):
        self._training_array = self._extract_data[
                                :int(self._training_data_ratio * len(self._extract_data))
                               ]
        self._testing_array = self._extract_data[
                               int(self._training_data_ratio * len(self._extract_data)) - self._training_window_size:
                              ]

        self._training_set_df = self._input_df[:int(self._training_data_ratio * len(self._extract_data))]
        self._testing_set_df = self._input_df[int(self._training_data_ratio * len(self._extract_data)):]

    @staticmethod
    def scaling_one_d_array(input_array: np.ndarray) -> (np.ndarray, MinMaxScaler):
        scaler = MinMaxScaler()
        scaler.fit(input_array.reshape(-1, 1))
        return scaler.transform(input_array.reshape(-1, 1)), scaler

    def _scaling_array(self):
        self._scaler_training.fit(self._training_array)
        self._scaler_testing.fit(self._testing_array)
        self._training_array = self._scaler_training.transform(self._training_array)
        self._testing_array = self._scaler_testing.transform(self._testing_array)

    def _preprocessing(self):
        # x_train, y_train = self._sliding_window_mask_on_training_data()
        # x_test, y_test = self._sliding_window_mask_on_testing_data()

        x_train, y_train = self._sliding_window_mask(self._training_array)
        x_test, y_test = self._sliding_window_mask(self._testing_array)

        # Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)
        # Reshape the data
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        self._training_tensor = torch.from_numpy(x_train).float()
        self._training_target_tensor = torch.from_numpy(y_train).float()

        self._testing_tensor = torch.from_numpy(x_test).float()
        self._testing_target_tensor = torch.from_numpy(y_test).float()

    def process_data(self) -> None:
        self._extract_training_data_and_scale()
        self._splitting()
        # self._scaling_array()
        self._preprocessing()

    def get_training_set(self) -> pd.DataFrame:
        return self._training_set_df

    def get_testing_set(self) -> pd.DataFrame:
        return self._testing_set_df

    def get_tensors_to_train_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._training_tensor, self._training_target_tensor

    def get_tensors_to_test_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._testing_tensor, self._testing_target_tensor

    def inverse_testing_scaler(self, data: np.ndarray, scaler_by_column_name) -> np.ndarray:
        return self._scaler_by_column[scaler_by_column_name].inverse_transform(data)
        # return self._scaler_testing.inverse_transform(data.reshape(-1, 1))

    def _sliding_window_mask(self, input_array: np.ndarray) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Sliding window mask on provided data, return training data and target data,
        """
        x = []
        y = []

        for i in range(self._training_window_size, len(input_array) - self._target_window_size + 1):
            x.append(input_array[i - self._training_window_size:i])
            y.append(input_array[i:i + self._target_window_size, 0])

        return x, y

    # def _sliding_window_mask_on_training_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    #     """
    #     Sliding window mask on training data, return training data and target data,
    #     the size of training data and the target data can be customized by window_size.
    #     """
    #
    #     x_train = []
    #     y_train = []
    #
    #     for i in range(self._training_window_size, len(self._training_array) - self._target_window_size + 1):
    #         x_train.append(self._training_array[i - self._training_window_size:i])
    #         y_train.append(self._training_array[i:i + self._target_window_size, 0])
    #
    #     return x_train, y_train
    #
    # def _sliding_window_mask_on_testing_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    #     x_test = []
    #     y_test = []
    #
    #     for i in range(self._training_window_size, len(self._testing_array) - self._target_window_size + 1):
    #         x_test.append(self._testing_array[i - self._training_window_size:i])
    #         y_test.append(self._testing_array[i:i + self._target_window_size, 0])
    #
    #     print(self._training_window_size, len(self._testing_array) - self._target_window_size + 1)
    #
    #     return x_test, y_test

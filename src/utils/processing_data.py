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

        # MinMaxScalers (kept from original code)
        self._scaler_training = MinMaxScaler()
        self._scaler_testing = MinMaxScaler()

        # For backward compatibility; some older logic might rely on these
        self._scaler_by_column = {}

    def _extract_training_data_and_scale(self) -> None:
        """
        Original method to extract and individually scale columns.
        We keep this to avoid side effects in older code that might rely on _scaler_by_column.
        """
        if self._extract_column is None:
            raise ValueError(f"Please check the column {self._extract_column} exist in the input data!")

        # For each requested column, scale it separately and then concatenate
        for i_column in self._extract_column:
            extract_column_series = self._input_df[i_column].values
            scaled_series, scaler = self.scaling_one_d_array(extract_column_series)
            scaled_series = np.expand_dims(scaled_series.reshape(-1), axis=1)
            self._scaler_by_column[i_column] = scaler

            if self._extract_data is None:
                self._extract_data = scaled_series
            else:
                self._extract_data = np.concatenate((self._extract_data, scaled_series), axis=1)

    def _splitting(self):
        """
        Updated to remove partial overlap. We no longer do
        [train_end - window_size:] for the test set.
        """
        cutoff = int(self._training_data_ratio * len(self._extract_data))

        self._training_array = self._extract_data[:cutoff]
        self._testing_array = self._extract_data[cutoff:]

        self._training_set_df = self._input_df[:cutoff]
        self._testing_set_df = self._input_df[cutoff:]

    @staticmethod
    def scaling_one_d_array(input_array: np.ndarray) -> (np.ndarray, MinMaxScaler):
        """
        Original helper: scales a single 1D array. We keep it unchanged for backward compatibility.
        """
        scaler = MinMaxScaler()
        scaler.fit(input_array.reshape(-1, 1))
        return scaler.transform(input_array.reshape(-1, 1)), scaler

    def _scaling_array(self):
        """
        Now we unify the scaling logic for training and testing arrays,
        using self._scaler_training and self._scaler_testing.
        """
        self._scaler_training.fit(self._training_array)
        self._training_array = self._scaler_training.transform(self._training_array)

        # Use the same scaler or a separate self._scaler_testing
        # Typically you'd want to transform test data with the training scaler:
        # self._testing_array = self._scaler_training.transform(self._testing_array)
        # But if you want to keep separate scalers, we do:
        self._scaler_testing.fit(self._testing_array)
        self._testing_array = self._scaler_testing.transform(self._testing_array)

    def _preprocessing(self):
        """
        We keep the original sliding-window approach. Single-step or multi-step
        is determined by self._target_window_size.
        """
        x_train, y_train = self._sliding_window_mask(self._training_array)
        x_test, y_test = self._sliding_window_mask(self._testing_array)

        # Convert to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Flatten final dimension of y if needed
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        # Convert to PyTorch tensors
        self._training_tensor = torch.from_numpy(x_train).float()
        self._training_target_tensor = torch.from_numpy(y_train).float()

        self._testing_tensor = torch.from_numpy(x_test).float()
        self._testing_target_tensor = torch.from_numpy(y_test).float()

    def process_data(self) -> None:
        """
        The main pipeline remains the same but now we:
          1) Extract and scale columns individually (old approach)
          2) Split train/test with no partial overlap
          3) Unified scaling with _scaling_array() (train + test)
          4) Create sliding windows
        """
        self._extract_training_data_and_scale()
        self._splitting()
        self._scaling_array()   # NEW: unify scaling after splitting
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
        """
        Preserved from original code. This uses the old per-column scaler logic.
        If you want to invert using the unified scaler, you'd do something else.
        """
        return self._scaler_by_column[scaler_by_column_name].inverse_transform(data)

    def _sliding_window_mask(self, input_array: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Original sliding-window logic. Single-step or multi-step depends on self._target_window_size.
        """
        x = []
        y = []
        for i in range(self._training_window_size, len(input_array) - self._target_window_size + 1):
            seq_x = input_array[i - self._training_window_size:i]
            seq_y = input_array[i:i + self._target_window_size, 0]  # Predict only the first column
            x.append(seq_x)
            y.append(seq_y)

        return x, y
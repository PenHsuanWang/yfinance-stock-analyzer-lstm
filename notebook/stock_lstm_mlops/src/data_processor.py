import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

class TimeSeriesDataProcessor:
    def __init__(self, seq_length: int, train_split_ratio: float):
        self.seq_length = seq_length
        self.train_split_ratio = train_split_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform_scaler(self, data: np.ndarray):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return self.scaler.inverse_transform(data)

    def create_sequences(self, scaled_data: np.ndarray):
        X, y = [], []
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i - self.seq_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        # Reshape to (samples, time_steps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def get_train_test_split(self, dataset: np.ndarray, scaled_data: np.ndarray):
        training_data_len = int(np.ceil(len(dataset) * self.train_split_ratio))
        
        # Train data
        train_data = scaled_data[0:training_data_len, :]
        X_train, y_train = [], []
        for i in range(self.seq_length, len(train_data)):
            X_train.append(train_data[i - self.seq_length:i, 0])
            y_train.append(train_data[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Test data
        test_data = scaled_data[training_data_len - self.seq_length:, :]
        X_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(self.seq_length, len(test_data)):
            X_test.append(test_data[i - self.seq_length:i, 0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'training_data_len': training_data_len
        }
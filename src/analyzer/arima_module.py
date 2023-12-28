# src/analyzer/arima_module.py
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

class ARIMAModule:
    def __init__(self, series, start_p=1, start_q=1, max_p=5, max_q=5, m=12, seasonal=True):
        self.series = series
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.m = m
        self.seasonal = seasonal
        self.model = None
        self.fitted_values = None
        self.residuals = None

    def find_best_model(self):
        self.model = auto_arima(self.series, start_p=self.start_p, start_q=self.start_q,
                                max_p=self.max_p, max_q=self.max_q, m=self.m,
                                seasonal=self.seasonal, error_action='ignore',
                                suppress_warnings=True, stepwise=True)
        print(self.model.summary())

    def fit_model(self):
        if not self.model:
            raise ValueError("No model is found. Run find_best_model() first.")
        order = self.model.order
        self.fitted_model = ARIMA(self.series, order=order).fit()
        self.fitted_values = self.fitted_model.fittedvalues
        self.residuals = self.fitted_model.resid

    def plot_decomposition(self):
        decomposition = seasonal_decompose(self.series, model='additive')
        plt.figure(figsize=(14, 7))
        plt.subplot(411)
        plt.plot(decomposition.observed, label='Observed')
        plt.legend(loc='upper left')
        plt.title('Seasonal Decomposition')
        # ... (省略其他部分以節省空間)
        plt.tight_layout()
        plt.show()

    def check_stationarity(self):
        result = adfuller(self.series)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

    def plot_fitted_vs_original(self):
        plt.figure(figsize=(15, 7))
        plt.plot(self.series, label='Original Time Series')
        plt.plot(self.fitted_values, label='ARIMA Fitted Values', alpha=0.7)
        plt.legend()
        plt.title('Original vs Fitted Time Series')
        plt.show()


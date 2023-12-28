import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class GMMModule:
    def __init__(self, data, n_components_range=(1, 10)):
        self.data = data
        self.n_components_range = range(*n_components_range)
        self.model = None
        self.labels = None

    def fit_model(self):
        # compute feature
        self.data['fracChange'] = (self.data['Close'] - self.data['Open']) / self.data['Open']
        self.data['fracHigh'] = (self.data['High'] - self.data['Open']) / self.data['Open']
        self.data['fracLow'] = (self.data['Open'] - self.data['Low']) / self.data['Open']
        observations = self.data[['fracChange', 'Volume']].values

        # select the best component
        aics = []
        for n_components in self.n_components_range:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(observations)
            aics.append(gmm.aic(observations))
        best_n_components = self.n_components_range[np.argmin(aics)]

        # fit the model
        self.model = GaussianMixture(n_components=best_n_components)
        self.model.fit(observations)
        self.labels = self.model.predict(observations)

    def plot_results(self):
        plt.figure(figsize=(15, 8))
        plt.scatter(self.data.index, self.data['Close'].values, c=self.labels, cmap='viridis', marker='o')
        plt.title('GMM on Stock Data')
        plt.xlabel('Day')
        plt.ylabel('Close Price')
        plt.colorbar(label='GMM Component')
        plt.show()

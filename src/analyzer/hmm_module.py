import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

class HMMModule:
    def __init__(self, data, n_components=4):
        self.data = data
        self.n_components = n_components
        self.model = None
        self.hidden_states = None

    def fit_model(self):
        # compute feature
        self.data['fracChange'] = (self.data['Close'] - self.data['Open']) / self.data['Open']
        observations = self.data[['fracChange', 'Volume']].values

        # training HMM
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full")
        self.model.fit(observations)
        self.hidden_states = self.model.predict(observations)

    def plot_results(self):
        plt.figure(figsize=(15, 8))
        for i in range(self.model.n_components):
            state_mask = self.hidden_states == i
            plt.plot(self.data.index[state_mask], self.data['Close'][state_mask], '.', label=f'State {i}')
        plt.legend()
        plt.title('Hidden State Sequence')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.show()


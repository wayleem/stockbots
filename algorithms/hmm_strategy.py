import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class HMMStrategy:
    def __init__(self, n_components=3, n_iter=10000):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = None
        self.scaler = StandardScaler()

    def train(self, data):
        features = data[['Log_Returns', 'Volatility', 'Volume_Change']].values
        scaled_features = self.scaler.fit_transform(features)
        self.model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.n_iter, random_state=42)
        self.model.fit(scaled_features)

    def generate_signals(self, data):
        if self.model is None:
            self.train(data)
        
        features = data[['Log_Returns', 'Volatility', 'Volume_Change']].values
        scaled_features = self.scaler.transform(features)
        states = self.model.predict(scaled_features)
        
        signals = np.zeros_like(states)
        signals[1:] = np.sign(states[1:] - states[:-1])
        return signals

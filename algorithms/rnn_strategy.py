import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

class RNNStrategy:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data):
        features = ['Adj Close', 'Volume', 'Returns', 'Log_Returns', 'Volatility', 'Volume_Change']
        scaled_data = self.scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])  # Predicting the next day's Adj Close
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, data):
        X, y = self.prepare_data(data)
        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    def generate_signals(self, data):
        if self.model is None:
            self.train(data)
        
        X, _ = self.prepare_data(data)
        predictions = self.model.predict(X)
        
        signals = np.zeros(len(data))
        signals[self.lookback:] = np.where(predictions.flatten() > X[:, -1, 0], 1, -1)
        return signals

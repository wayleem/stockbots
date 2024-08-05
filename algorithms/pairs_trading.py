import numpy as np
import pandas as pd

class PairsStrategy:
    def __init__(self, window_size=20, zscore_threshold=1.0):
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold

    def calculate_zscore(self, ratio):
        return (ratio - ratio.rolling(window=self.window_size).mean()) / ratio.rolling(window=self.window_size).std()

    def generate_signals(self, data):
        # Assuming data contains 'Stock1' and 'Stock2' columns
        ratio = data['Stock1'] / data['Stock2']
        zscore = self.calculate_zscore(ratio)
        
        signals = pd.Series(0, index=data.index)
        signals[zscore > self.zscore_threshold] = -1  # Short the pair
        signals[zscore < -self.zscore_threshold] = 1  # Long the pair
        return signals

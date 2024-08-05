import pandas as pd

class MACDStrategy:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_macd(self, data):
        close = data['Adj Close']
        exp1 = close.ewm(span=self.fast_period, adjust=False).mean()
        exp2 = close.ewm(span=self.slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        return macd, signal

    def generate_signals(self, data):
        macd, signal = self.calculate_macd(data)
        signals = pd.Series(0, index=data.index)
        signals[macd > signal] = 1  # Buy signal
        signals[macd < signal] = -1  # Sell signal
        return signals

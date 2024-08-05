import pandas as pd

class Backtester:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital

    def run(self, data, signals):
        portfolio = pd.DataFrame(index=data.index).fillna(0.0)
        portfolio['holdings'] = signals * data['Adj Close']
        portfolio['cash'] = self.initial_capital - (signals.diff() * data['Adj Close']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

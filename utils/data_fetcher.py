import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    data['Log_Returns'] = np.log(data['Adj Close']).diff()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Volume_Change'] = data['Volume'].pct_change()
    return data.dropna()

import yfinance as yf
import pandas as pd
import numpy as np
from time import sleep
from requests.exceptions import ConnectionError

def fetch_data(ticker, start_date, end_date, max_retries=5):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                data['Returns'] = data['Adj Close'].pct_change()
                data['Log_Returns'] = np.log(data['Adj Close']).diff()
                data['Volatility'] = data['Returns'].rolling(window=20).std()
                data['Volume_Change'] = data['Volume'].pct_change()
                return data.dropna()
            else:
                print(f"Attempt {attempt + 1}: No data retrieved. Retrying...")
        except ConnectionError as e:
            print(f"Attempt {attempt + 1}: Connection error occurred. Retrying in 5 seconds...")
            sleep(5)
    
    print(f"Failed to fetch data after {max_retries} attempts.")
    return pd.DataFrame()  # Return an empty DataFrame if all attempts fail

import argparse
from algorithms.hmm_strategy import HMMStrategy
from algorithms.macd_strategy import MACDStrategy
from algorithms.pairs_trading import PairsStrategy
from algorithms.rnn_strategy import RNNStrategy
from utils.data_fetcher import fetch_data
from utils.backtester import Backtester
from utils.performance_metrics import calculate_metrics
from config import CONFIG

def main():
    parser = argparse.ArgumentParser(description="Compare stock trading algorithms")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("start_date", help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date for backtesting (YYYY-MM-DD)")
    args = parser.parse_args()

    # Fetch data
    data = fetch_data(args.ticker, args.start_date, args.end_date)

    # Initialize strategies
    strategies = [
        HMMStrategy(n_components=CONFIG['HMM_COMPONENTS']),
        MACDStrategy(fast_period=CONFIG['MACD_FAST'], slow_period=CONFIG['MACD_SLOW'], signal_period=CONFIG['MACD_SIGNAL']),
        PairsStrategy(window_size=CONFIG['PAIRS_WINDOW'], zscore_threshold=CONFIG['PAIRS_ZSCORE_THRESHOLD']),
        RNNStrategy(lookback=CONFIG['LOOKBACK_PERIOD'])
    ]

    # Run backtests
    backtester = Backtester(initial_capital=CONFIG['INITIAL_BALANCE'])
    results = {}

    for strategy in strategies:
        signals = strategy.generate_signals(data)
        backtest_result = backtester.run(data, signals)
        metrics = calculate_metrics(backtest_result)
        results[strategy.__class__.__name__] = metrics

    # Print comparison
    print(f"Comparison results for {args.ticker} from {args.start_date} to {args.end_date}:")
    for strategy_name, metrics in results.items():
        print(f"\n{strategy_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()

import numpy as np

def calculate_metrics(portfolio):
    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
    sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std()
    max_drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1).min()
    
    return {
        'Total Return': f'{total_return:.2%}',
        'Sharpe Ratio': f'{sharpe_ratio:.2f}',
        'Max Drawdown': f'{max_drawdown:.2%}'
    }

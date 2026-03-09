import numpy as np
import pandas as pd

_PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "ME": 12, "Q": 4, "QE": 4, "A": 1, "Y": 1, "YE": 1}


def calculate_performance(returns: pd.Series, freq: str = "M") -> dict:
    """
    Calculate CAGR, Sharpe ratio, and max drawdown.

    Parameters
    ----------
    returns : pd.Series
        Periodic return series (e.g. monthly).
    freq : str
        Pandas frequency alias of the return series ('M', 'D', 'Q', etc.).
        Used to derive the annualisation factor.
    """
    periods = _PERIODS_PER_YEAR.get(freq, 252)

    returns = returns.dropna()

    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1]
    n_years = len(returns) / periods

    cagr = total_return ** (1 / n_years) - 1

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(periods)
        if returns.std() != 0
        else 0
    )

    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max) - 1
    max_drawdown = drawdown.min()

    return {
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 4),
        "MaxDrawdown": round(max_drawdown, 4),
    }


# Alias used by main.py
performance_summary = calculate_performance

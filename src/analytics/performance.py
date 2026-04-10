import numpy as np
import pandas as pd

_PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "ME": 12, "Q": 4, "QE": 4, "A": 1, "Y": 1, "YE": 1}


def calculate_performance(returns: pd.Series, freq: str = "M") -> dict:
    """
    Calculate absolute performance metrics.

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
    n_years = len(returns) / periods
    cagr = cumulative.iloc[-1] ** (1 / n_years) - 1

    volatility = returns.std() * np.sqrt(periods)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(periods)
        if returns.std() != 0
        else 0.0
    )

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    sortino = (
        returns.mean() / downside_std * np.sqrt(periods)
        if downside_std != 0
        else 0.0
    )

    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max) - 1
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    hit_rate = (returns > 0).mean()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    return {
        "CAGR": round(cagr, 4),
        "Volatility": round(volatility, 4),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "Calmar": round(calmar, 4),
        "MaxDrawdown": round(max_drawdown, 4),
        "HitRate": round(hit_rate, 4),
        "AvgWin": round(avg_win, 4),
        "AvgLoss": round(avg_loss, 4),
    }


def calculate_relative_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    freq: str = "M",
) -> dict:
    """
    Compute benchmark-relative metrics: excess CAGR, beta, tracking error,
    and information ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Net periodic portfolio return series.
    benchmark_returns : pd.Series
        Benchmark periodic return series (same frequency).
    freq : str
        Frequency alias used to annualise ('M', 'D', 'Q', etc.).
    """
    periods = _PERIODS_PER_YEAR.get(freq, 252)

    port, bench = portfolio_returns.align(benchmark_returns, join="inner")
    port, bench = port.dropna(), bench.dropna()
    port, bench = port.align(bench, join="inner")

    active = port - bench

    bench_cagr = (1 + bench).prod() ** (periods / len(bench)) - 1
    port_cagr = (1 + port).prod() ** (periods / len(port)) - 1
    excess_cagr = port_cagr - bench_cagr

    beta = port.cov(bench) / bench.var() if bench.var() != 0 else float("nan")

    tracking_error = active.std() * np.sqrt(periods) if active.std() != 0 else 0.0

    information_ratio = (
        active.mean() / active.std() * np.sqrt(periods)
        if active.std() != 0
        else 0.0
    )

    return {
        "BenchmarkCAGR": round(bench_cagr, 4),
        "ExcessCAGR": round(excess_cagr, 4),
        "Beta": round(beta, 4),
        "TrackingError": round(tracking_error, 4),
        "InformationRatio": round(information_ratio, 4),
    }


def rolling_sharpe(returns: pd.Series, window: int = 12, freq: str = "M") -> pd.Series:
    """
    Compute rolling annualised Sharpe ratio over a trailing window.

    Parameters
    ----------
    window : int
        Number of periods in the rolling window.
    freq : str
        Frequency alias used to annualise.
    """
    periods = _PERIODS_PER_YEAR.get(freq, 252)
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    result = roll_mean / roll_std * np.sqrt(periods)
    return result.where(roll_std != 0, other=0.0).rename("rolling_sharpe")


# Alias used by main.py
performance_summary = calculate_performance

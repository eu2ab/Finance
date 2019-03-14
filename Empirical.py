from __future__ import division
import numpy as np
import pandas as pd
from GetData import QUANDL_prices, getFRED
import statsmodels.api as sm


def ols_single(ticker, start, end):
    """
    Gives OLS summary table for security against S&P500

    :param ticker: e.g. ['AAPL']
    :param start: start date
    :param end: end date
    :return: OLS Summary table
    """
    r = QUANDL_prices(ticker, start, end).pct_change()  # Portfolio
    m = getFRED('SP500', start, end).pct_change()  # S&P500, market basket proxy
    merged = pd.concat([r, m], axis=1, join='inner')  # Inner join to make sure all have same number of rows
    merged = merged.dropna(axis=0)

    x = merged[m.columns]
    x = sm.add_constant(x)  # Manually add constant before placing in OLS model
    y = merged[r.columns]
    results = sm.OLS(y, x).fit()  # Fit model
    return (results.summary())


def calc_portfolio_var(df, weights=None):
    """
    Calculate portfolio variance

    :param df: Daily prices of portfolio of stocks
    :param weights: Weights of each security in the portfolio, if needed
    :return: Variance of portfolio, adjusted for weighting
    """
    df = df.pct_change()
    if weights is None:
        weights = np.ones(df.columns.size) / df.columns.size
    sigma = df.cov()
    var = (weights * sigma * weights.T).sum()
    return (var)


def calc_portfolio_cov(df):
    """
    Calculate portfolio covariance

    :param df: Daily prices of portfolio of stocks
    :return: Covariance of the portfolio, adjusted for weighting
    """
    df = df.pct_change()
    return (df.cov())


def calc_portfolio_mean(df, weights=None):
    """
    Calculate portfolio average

    :param df: Daily prices of portfolio of stocks
    :param weights: Weights of each security in the portfolio, if needed
    :return: Average of the portfolio, adjusted for weighting
    """
    df = df.pct_change()
    if weights is None:
        weights = np.ones(df.columns.size) / df.columns.size
    mean = df.mean()
    mean = (weights * mean * weights.T).sum()
    return (mean)


def sharpe_ratio(df, weights=None, risk_free_rate=0.015):
    """
    Calculat sharpe ratio of the portfolio

    :param df: Daily prices of portfolio of stocks
    :param weights: Weighting of securities, if applicable
    :param risk_free_rate: Risk free rate, w/ default of 1.5%
    :return: Sharpe ratio of the portfolio, adjusted for weighting if applicable
    """
    n = df.columns.size
    if weights is None:
        weights = np.ones(n) / n

    var = calc_portfolio_var(df, weights)  # get the portfolio variance
    means = df.pct_change().mean()  # and the means of the stocks in the portfolio
    return ((means.dot(weights) - risk_free_rate) / np.sqrt(var))  # and return the sharpe ratio


def index_returns(df, starting_value=0):
    """
    Index returns starting from a starting value

    :param df: Daily prices of portfolio of stocks
    :param starting_value: Default = 0, but can be 100 or something
    :return: Indexed returns
    """
    if len(df) < 1:
        return np.nan

    df_cum = np.exp(np.log1p(df.pct_change()).cumsum())

    if starting_value == 0:
        return (df_cum - 1)
    else:
        return (df_cum * starting_value)


def port_beta(df, spy):
    """
    Returns the beta of daily returns over a certain period of time; market portfolio is the daily returns of SPY
    over same period of time as the main stock basket

    :param df: Daily prices of portfolio of stocks
    :param spy: Daily prices of SPY etf
    :return: Portfolio beta
    """
    df = df.pct_change()
    market = spy.pct_change()
    m = np.matrix([df, market])
    return (np.cov(m)[0][1] / np.std(market))


def hist_var(df, notionalvalue=10000, weights=None, alpha=0.05, lookback_days=252):
    """
    Historical method of solving for Value at Risk

    :param df: Daily prices of portfolio
    :param notionalvalue: Notional value of starting portfolio
    :param weights: Weights of portfolio
    :param alpha: Level of significance
    :param lookback_days: Lookback period
    :return: Notional value at risk (divide by total notional value to get percentage)
    """
    lookback_days = min(len(df), lookback_days)
    if weights is None:
        weights = np.ones(df.columns.size) / df.columns.size

    returns = df.fillna(0.0)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    return (np.percentile(portfolio_returns, 100 * alpha) * notionalvalue)


def hist_cvar(df, notionalvalue=10000, weights=None, alpha=0.05, lookback_days=252):
    """
    Historical method for solving conditional VAR (or expected shortfall)

    :param df: Daily prices of portfolio
    :param notionalvalue: Notional value of portfolio
    :param weights: Weights of portfolio
    :param alpha: Level of significance
    :param lookback_days: Lookback period
    :return: Notional CVAR (divide by total notional value to get percentage)
    """
    lookback_days = min(len(df), lookback_days)
    if weights is None:
        weights = np.ones(df.columns.size) / df.columns.size

    var = hist_var(df, notionalvalue, weights, alpha, lookback_days)
    returns = df.fillna(0.0)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    var_pct_loss = var / notionalvalue
    return (notionalvalue * np.nanmean(portfolio_returns[portfolio_returns < var_pct_loss]))

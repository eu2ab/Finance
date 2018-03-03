import pandas as pd


def ma(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of moving average added to df, w/ n being the # of days
    """
    df = pd.rolling_mean(df, n)
    return df


def ema(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of exponential moving average added to df, w/ n being the # of days
    """
    df = pd.ewma(df, span=n, min_periods=n - 1)
    return df


def mom(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of momentum added to df, w/ n being the # of days
    """
    df = df.diff(n)
    return df


def roc(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of rate of change added to df, w/ n being the # of days
    """
    m = df.diff(n - 1)
    n = df.shift(n - 1)
    df = m / n
    return df


def bbands(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of bollinger bands added to df, w/ n being the # of days
    """
    _ma = pd.rolling_mean(df, n)
    msd = pd.rolling_std(df, n)
    b1 = 4 * msd / _ma
    b2 = (df - _ma + 2 * msd) / (4 * msd)
    return b1, b2


def macd(df, n_fast, n_slow):
    emafast = pd.ewma(df['Close'], span=n_fast, min_periods=n_slow - 1)
    emaslow = pd.ewma(df['Close'], span=n_slow, min_periods=n_slow - 1)
    _macd = emafast - emaslow
    macdsign = pd.ewma(_macd, span=9, min_periods=8)
    macddiff = _macd - macdsign
    return _macd, macdsign, macddiff


def rolling_sigma(df, n):
    """
    :param df: dataframe from historical_prices, found under the 'Stocks' module
    :param n: number of days
    :return: add rolling standard deviation to df
    """
    df = pd.rolling_std(df, n)
    return df

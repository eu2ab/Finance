import numpy as np
import pandas as pd
import math as m
from matplotlib import pyplot as plt
import datetime

def MA(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of moving average added to df, w/ n being the # of days
    """
    df = pd.rolling_mean(df, n)
    return df

def EMA(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of exponential moving average added to df, w/ n being the # of days
    """
    df = pd.ewma(df, span = n, min_periods = n - 1)
    return df

def MOM(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of momentum added to df, w/ n being the # of days
    """
    df = df.diff(n)
    return df

def ROC(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of rate of change added to df, w/ n being the # of days
    """
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    df = M / N
    return df

def BBANDS(df, n):
    """
    :param df: dataframe of closing prices
    :param n: number of days
    :return: another column of bollinger bands added to df, w/ n being the # of days
    """
    MA = pd.rolling_mean(df, n)
    MSD = pd.rolling_std(df, n)
    b1 = 4 * MSD / MA
    b2 = (df - MA + 2 * MSD) / (4 * MSD)
    return b1, b2

def MACD(df, n_fast, n_slow):
    EMAfast = pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1)
    EMAslow = pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1)
    MACD = EMAfast - EMAslow
    MACDsign = pd.ewma(MACD, span = 9, min_periods = 8)
    MACDdiff = MACD - MACDsign
    return MACD, MACDsign, MACDdiff

def TSI(df, r, s):
    """
    :param df: dataframe from historical_prices, found under the 'Stocks' module
    :param r:
    :param s:
    :return: True Strength Indicator added to df
    """
    M = df.diff(1)
    aM = abs(M)
    EMA1 = pd.ewma(M, span = r, min_periods = r - 1)
    aEMA1 = pd.ewma(aM, span = r, min_periods = r - 1)
    EMA2 = pd.ewma(EMA1, span = s, min_periods = s - 1)
    aEMA2 = pd.ewma(aEMA1, span = s, min_periods = s - 1)
    df = EMA2 / aEMA2
    return df

def RollingSigma(df, n):
    """
    :param df: dataframe from historical_prices, found under the 'Stocks' module
    :param n: number of days
    :return: add rolling standard deviation to df
    """
    df = pd.rolling_std(df, n)
    return df
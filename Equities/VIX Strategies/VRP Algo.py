from urllib.request import urlopen
import pandas as pd
from pandas.io.data import DataReader
import datetime as dt
import numpy as np
# http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html for more visualizations

# Yahoo Finance API no longer works. VIX and VXV do not have QUANDL support thus QUANDL_price does not work.
# This algorithm is thus broken.

def historical_close(ticker, _start, _end):     # datetime(2000, 1, 1) ***(yyyy, m, d) format****
    x = DataReader(ticker, 'yahoo', _start, _end)
    return x['Adj Close']

# set parameters for query
N = 150  # number of days into the past
end = dt.date.today()
start = end - dt.timedelta(days=N)
stocks = ['^VIX', '^VXV', 'XIV', 'VXX']

# import securities
df = historical_close(stocks, _start=start, _end=end)

# build dictionary
_dict = pd.Series.to_dict(df)

# specific stocks
VIX = _dict['^VIX']  # VIX
VXV = _dict['^VXV']  # VXV
XIV = _dict['XIV']   # XIV
VXX = _dict['VXX']   # VXX
spread_3m = VXV - VIX   # spread between 3m and spot
Vratio = VXV/VIX    # above 1 = contango, below 1 = backwardation
slnVIX = np.std(np.log(VIX))    # standard deviation of log of VIX


# graphs
# Vratio.plot() graph of Vratio
df = pd.concat([VXV, VIX], axis=1)  # graph of VXV and VIX
df.plot(legend=True)    # include legend
df.plot(subplots=True)  # using subplots
df.plot.area(stacked=False)

# using a secondary y axis
VIX.plot()
Vratio.plot(secondary_y=True)   # must be done after first graph

pd.rolling_corr(VXX, XIV, 15)     # rolling correlation with 15 day window
df.corr()   # for correlation matrix of all stocks in df portfolio
from Empyrical import daily_percentage_change
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

#plotting a correlation TS of two securities
#retrieving the data
ticker = ['MSFT','TLT']
start = datetime(2008,1,1)
end = datetime(2016,1,1)
returns = daily_percentage_change(ticker, start, end)

#plotting both time series
returns = returns.cumsum()
returns.plot()

#calculating rolling correlations
correls = pd.rolling_corr(returns, 50) #50 day rolling correlation
correls.ix[:,'MSFT','TLT'].plot() #PICK ANY TWO SECURITIES from ticker
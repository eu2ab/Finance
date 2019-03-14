from GetData import QUANDL_price_pct_change
import pandas as pd
from datetime import datetime

# Plotting a correlation TS of two securities
# Retrieving the data
ticker = ['MSFT', 'TLT']
start = datetime(2008, 1, 1)
end = datetime(2016, 1, 1)
returns = QUANDL_price_pct_change(ticker, start, end)

# Plotting both time series
returns = returns.cumsum()
returns.plot()

# Calculating rolling correlations
correls = pd.rolling_corr(returns, 50)  # 50 day rolling correlation
correls.ix[:, 'MSFT', 'TLT'].plot()   # PICK ANY TWO SECURITIES from ticker

from GetData import QUANDL_price_pct_change
import pandas as pd
from datetime import datetime

# plotting a correlation TS of two securities
# retrieving the data
ticker = ['MSFT', 'TLT']
start = datetime(2008, 1, 1)
end = datetime(2016, 1, 1)
returns = QUANDL_price_pct_change(ticker, start, end)

# plotting both time series
returns = returns.cumsum()
returns.plot()

# calculating rolling correlations
correls = pd.rolling_corr(returns, 50)  # 50 day rolling correlation
correls.ix[:, 'MSFT', 'TLT'].plot()   # PICK ANY TWO SECURITIES from ticker

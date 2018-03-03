from GetData import QUANDL_price
import pandas as pd
import datetime

# setting constraints
stocks = ['tlt', 'spy']     # example group of stocks
start = datetime.date(2016, 11, 8)
end = datetime.date(2016, 12, 14)

# query data
df = QUANDL_price(stocks, start, end)
# build dictionary
_dict = pd.Series.to_dict(df)

# if wanting to plot full universe, use df and not dict
df.plot()   # shows universe
_dict['googl'].plot()    # picks a specific stock

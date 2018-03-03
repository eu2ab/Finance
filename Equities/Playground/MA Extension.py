"""
Brings in all stocks in S&P500 and search for extreme upward extensions from their respective moving average
(50 day MA by default).
"""

from Finance.STOCKS.Stocks import *

# full universe of S&P500
df = SP500_df(datetime.date(2016, 6, 6), datetime.date.today(), 'close')

# moving average of S&P500
df1 = pd.DataFrame()
for i in range(0, len(df.columns)):
    df2 = pd.rolling_mean(df.ix[:, i], 50)  # 50 day moving average
    df1 = df1.append(df2)

# find difference between df and df1, on the last row (trading day)
df_last = df.ix[-1, :]  # normal price on last trading day
df1_last = df1.ix[:, -1]  # 50MA on last trading day

# divide to find percentage difference as opposed to nominal difference
delta = df_last/df1_last  # higher number indicates larger upward divergences from  50MA..what to short
delta = delta.sort_values(ascending=False)  # sorting delta from largest to smallest
delta.head()  # looking at only the top diverging results

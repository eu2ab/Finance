from Stocks import historical_close, historical_prices
import pandas as pd
import datetime as datetime

#setting constraints
stocks = ['^VIX', 'spy'] #example group of stocks
start = datetime.date(2016,11,8)
end = datetime.date(2016,12,14)

#query data
df = historical_close(stocks,start,end)
#build dictionary
dict = pd.Series.to_dict(df)

#if wanting to plot full universe, use df and not dict
df.plot() #shows universe
dict['googl'].plot() #picks a specific stock
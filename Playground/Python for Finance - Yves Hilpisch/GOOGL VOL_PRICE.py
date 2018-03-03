from GetData import QUANDL_price
import numpy as np
import pandas as pd
import datetime


googl = QUANDL_price(['GOOGL'], datetime.date(2010, 1, 1), datetime.date.today())
googl['Log_Ret'] = np.log(googl / googl.shift(1))   # calculate log returns
googl['Volatility'] = pd.rolling_std(googl['Log_Ret'], window=252) * np.sqrt(252)    # calculating volatility

# plotting
googl[['GOOGL', 'Volatility']].plot(subplots=True, color='blue', figsize=(8, 6))

# produces a graph of GOOGL closing prices and volatility at the bottom

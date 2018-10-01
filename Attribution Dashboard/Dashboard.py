import numpy as np
import pandas as pd
import datetime as dt
from iexfinance import get_historical_data

#   Tickers
ticks = {'AAPL': 15,
         'GOOGL': 15,
         'TSLA': 15}  # edit tickers and delta values here
start = dt.datetime.now() - dt.timedelta(days=5 * 365)  # five years back from today
end = dt.datetime.now()  # today

#   Build data frame containing positions/information
df = pd.DataFrame({'Ticker': list(ticks.keys()),
                   'Cost Basis': [100, 100, 100],
                   'Date of Purchase': [dt.date(2016, 1, 1), dt.date(2016, 1, 1), dt.date(2016, 1, 1)],
                   'Delta': list(ticks.values())})  # Build initial dataframe

prices = get_historical_data(list(ticks.keys()), start=start, end=end, output_format='pandas')  # pull data from IEX
port_prices = pd.DataFrame()  # convert to just dataframe based on closed prices
for key in prices:
    port_prices[key] = prices[key]['close']
spy = get_historical_data('SPY', start=start, end=end, output_format='pandas')['close']

df['Stock Price'] = port_prices.tail(1).transpose().values
df['Unrlzd Gain/Loss'] = (df['Stock Price'] - df['Cost Basis']) * df['Delta']
df['Portfolio Weights'] = (df['Stock Price'] * df['Delta'] / np.inner(df['Stock Price'].values, df['Delta']))

#   Calculate rolling weights
values_df = pd.DataFrame()  # values of securities
for key in ticks:
    values_df[key] = port_prices[key] * ticks[key]

weights_df = pd.DataFrame()  # weights of securities based on sum of values
for key in ticks:
    weights_df[key] = values_df[key] / values_df.sum(axis=1)

# Building price change datasets
wtd_prc_chng = weights_df * port_prices.pct_change() * 100  # daily weighted price change
wtd_port_chng = wtd_prc_chng.sum(axis=1)  # sum of weighted daily price changes
spy_chng_df = spy.pct_change() * 100  # daily price change of S&P500

# corr/std/var stuff
port_corr = pd.rolling_corr(wtd_port_chng, spy_chng_df, window=200)  # rolling portfolio corr v. S&P500
# port_corr = pd.rolling_corr(wtd_prc_chng,spy_chng_df, window=200) # rolling corr by stock
beta = np.cov(np.matrix([wtd_port_chng, spy_chng_df]))[0][0] / np.std(spy_chng_df)
sharpe = ((wtd_port_chng.mean() * 252) - 3) / (wtd_port_chng.std() * np.sqrt(252))  # annualized

#   ytd
port_ytd = wtd_port_chng.loc['2018-01-01':].cumsum()
spy_ytd = spy_chng_df.loc['2018-01-01':].cumsum()
excess_ytd = port_ytd.tail(1) - spy_ytd.tail(1)

#   last 3 months
port_3m = wtd_port_chng.iloc[-91:].cumsum()  # 91 days is roughly the last 3 months
spy_3m = spy_chng_df.iloc[-91:].cumsum()
excess_3m = port_3m.tail(1) - spy_3m.tail(1)

#   Modern Portfolio theory
weights = np.random.random(len(ticks))
weights /= np.sum(weights)
exp_ret = np.sum(port_prices.pct_change().mean() * weights) * 252  # expected portfolio return based on
# random weights

exp_var = np.dot(weights.T, np.dot(port_prices.pct_change().cov() * 252, weights))  # expected portfolio variance
exp_sd = np.sqrt(exp_var)

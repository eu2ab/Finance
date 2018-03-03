from GetData import QUANDL_price_pct_change
from Empirical import port_beta
import datetime
import pandas as pd

# Deprecated because QUANDL_price_pct_change does not support SPY. Original Yahoo Fiannce function
# created to use SPY has been deprecated.

start = datetime.date(2016, 1, 1)
end = datetime.date.today()

stocks = pd.DataFrame(['GS', 'TLT', 'HYG'], columns=['Securities'])
weights = pd.DataFrame([.5, .25, .25], columns=['Portfolio Weights'])
df = pd.concat([stocks, weights], axis=1)


def beta_table(stocks_weights, start, end):
    """
    Returns an attribution table for portfolio beta
    :param stocks_weights: table of the name of equities in one column, weights in another
    :param start: start date
    :param end: end date
    :return: table
    """
    # due to the nature of the percentchange function, first data point is always NaN, get around this
    market = QUANDL_price_pct_change('SPY', start, end)
    market = market[1:]

    betas = []
    weighted = []
    for i in range(0, len(df)):
        stock = QUANDL_price_pct_change(df.iat[i, 0], start, end)
        stock = stock[1:] * 100
        beta = port_beta(stock, market)
        betas.append(beta)
        weighted_beta = beta * df.iat[i, 1]
        weighted.append(weighted_beta)
    betas = pd.DataFrame(betas, columns=['Beta'])
    weighted = pd.DataFrame(weighted, columns=['Weighted Beta'])
    final = pd.concat([df, betas, weighted], axis=1)
    return final

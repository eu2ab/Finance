import numpy as np
import scipy as sp
import scipy.optimize as scopt
import datetime
import matplotlib.pyplot as plt
from GetData import QUANDL_price_pct_change


# MAX SHARPE PORTFOLIO 1 w/ MSFT, AAPL, and KO ####
ticker = ['aapl', 'msft', 'ko']     # use multiple stocks here
start_date = datetime.date(2000, 1, 1)
end_date = datetime.date(2016, 1, 1)
# Importing Data in NP format

daily_returns = QUANDL_price_pct_change(ticker, start_date, end_date)

def calc_annual_returns(daily_returns):
    """
    :param daily_returns: Daily percentage changes of dataframe of stocks
    :return: annual returns based on daily percentage returns
    """
    grouped = np.exp(daily_returns.groupby(
        lambda date: date.year).sum()) - 1
    return grouped

returns = calc_annual_returns(daily_returns)


def calc_portfolio_var(returns, weights=None):
    if weights is None:
        weights = np.ones(returns.columns.size) / \
        returns.columns.size
    sigma = np.cov(returns.T, ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var


def sharpe_ratio(returns, weights=None, risk_free_rate=0.015):
    n = returns.columns.size
    if weights is None:
        weights = np.ones(n)/n
    # get the portfolio variance
    var = calc_portfolio_var(returns, weights)
    # and the means of the stocks in the portfolio
    means = returns.mean()
    # and return the sharpe ratio
    return (means.dot(weights) - risk_free_rate)/np.sqrt(var)


def negative_sharpe_ratio_n_minus_1_stock(weights, returns, risk_free_rate):
    weights2 = sp.append(weights, 1 - np.sum(weights))
    return -sharpe_ratio(returns, weights2, risk_free_rate)


def optimize_portfolio(returns, risk_free_rate):
    # start with equal weights
    w0 = np.ones(returns.columns.size-1,
                 dtype=float) * 1.0 / returns.columns.size
    # minimize the negative sharpe value
    w1 = scopt.fmin(negative_sharpe_ratio_n_minus_1_stock,
                    w0, args=(returns, risk_free_rate))
    # build final set of weights
    final_w = sp.append(w1, 1 - np.sum(w1))
    # and calculate the final, optimized, sharpe ratio
    final_sharpe = sharpe_ratio(returns, final_w, risk_free_rate)
    return final_w, final_sharpe

# Visualize Efficient Frontier
target_ret = 0.10   # or something


def objfun(weights, returns, target_ret):
    stock_mean = np.mean(returns, axis=0)
    port_mean = np.dot(weights, stock_mean)     # port mean
    cov = np.cov(returns.T)     # var-cov matrix
    port_var = np.dot(np.dot(weights, cov), weights.T)        # port variance
    penalty = 2000 * abs(port_mean - target_ret)
    return np.sqrt(port_var) + penalty      # objective function


def calc_efficient_frontier(returns):
    result_means = []
    result_stds = []
    result_weights = []

    means = returns.mean()
    min_mean, max_mean = means.min(), means.max()
    nstocks = returns.columns.size

    for r in np.linspace(min_mean, max_mean, 100):
        weights = np.ones(nstocks)/nstocks
        bounds = [(0,1) for i in np.arange(nstocks)]
        constraints = ({'type': 'eq',
                       'fun': lambda w: np.sum(w) - 1})
        results = scopt.minimize(objfun, weights, (returns, r),
                                 method='SLSQP',
                                 constraints=constraints,
                                 bounds=bounds)
        if not results.success:
            raise Exception(results.message)
        result_means.append(np.round(r,4))
        std_=np.round(np.std(np.sum(returns*results.x,axis=1)),6)
        result_stds.append(std_)

        result_weights.append(np.round(results.x, 5))
    return {'Means': result_means,
            'Standard Deviations': result_stds,
            'Weights': result_weights}
# calc
frontier_data = calc_efficient_frontier(returns)


# plot
def plot_efficient_frontier(frontier_data):
    plt.figure(figsize=(12, 8))
    plt.title('Efficient Frontier')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Return')
    plt.plot(frontier_data['Standard Deviations'], frontier_data['Means'], '--')

# MAX SHARPE: 0.57

# MAX SHARPE PORTFOLIO 2 w/ MSFT, AAPL, and KO ####
r = returns.T
c = np.cov(returns.T, ddof=0)
rf = 0.015  # 1.5% default


def solve_weights(r, c, rf):
    def fitness(w, r, c, rf):
        # calculate mean/variance of the portfolio
        stock_mean = np.mean(r, axis=1)
        mean = np.dot(w, stock_mean)
        var = calc_portfolio_var(r)
        util = (mean - rf) / np.sqrt(var)      # utility = Sharpe ratio
        return 1/util                       # maximize the utility
    n = len(r)
    w = np.ones([n])/n                     # start with equal weights
    b_ = [(0., 1.) for i in range(n)]    # weights between 0%..100%. No leverage, no shorting
    c_ = ({'type': 'eq', 'fun': lambda w: sum(w)-1.})   # Sum of weights = 100%
    optimized = sp.optimize.minimize(fitness, w, (r, c, rf),
                method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

w=solve_weights(r, c, rf)
sharpe_ratio(returns, w, rf)    # .523

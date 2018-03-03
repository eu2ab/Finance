## European call options pricing with BSM, including vega and implied volatility estimation

def bsm_call_value(S0, K, T, r, sigma):
    """
    Valuation of European Call option in BSM model
    :param S0: float; initial stock level
    :param K: float; strike price
    :param T: float; maturity date (in year fractions)
    :param r: float; constant risk-free rate
    :param sigma: float; volatility factor in diffusion term
    :return: PV of european call option
    """
    from math import log, sqrt, exp
    from scipy import stats
    S0 = float(S0)
    d1 = (log(S0/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    # stats.norm.cdf is a cumulative distribution function
    return value

# Vega Function
def bsm_vega(S0, K, T, r, sigma):
    """
    Vega function of European option in BSM
    :param S0: float; spot
    :param K: float; strike
    :param T: float; maturity
    :param r: float; short term interest rate
    :param sigma: volatility factor
    :return: partial derivative of BSM w.r.t. sigma (i.e. Vega)
    """
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0/K) + (r + 0.5 * sigma **2)*T )/ (sigma * sqrt(T))
    vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
    return vega

# Implied Volatility Function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it = 100):
    """
    Implied volatility of European Call option in BSM model
    :param S0: float; spot
    :param K: float; strike
    :param T: float; maturity
    :param r: float; risk free rate
    :param C0: float; call option price?
    :param sigma_est: float; estimate of implied volatility
    :param it: integer; number of iterations
    :return: sigma_est; float of estimated implied volatility
    """
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)/(bsm_vega(S0, K, T, r, sigma_est)))

    return sigma_est


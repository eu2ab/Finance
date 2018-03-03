# European call options pricing with BSM, including vega and implied volatility estimation


def bsm_call_value(s0, k, t, r, sigma):
    """
    Valuation of European Call option in BSM model
    :param s0: float; initial stock level
    :param k: float; strike price
    :param t: float; maturity date (in year fractions)
    :param r: float; constant risk-free rate
    :param sigma: float; volatility factor in diffusion term
    :return: PV of european call option
    """
    from math import log, sqrt, exp
    from scipy import stats
    s0 = float(s0)
    d1 = (log(s0/k) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = (log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    value = (s0 * stats.norm.cdf(d1, 0.0, 1.0) - k * exp(-r * t) * stats.norm.cdf(d2, 0.0, 1.0))
    # stats.norm.cdf is a cumulative distribution function
    return value


# Vega Function
def bsm_vega(s0, k, t, r, sigma):
    """
    Vega function of European option in BSM
    :param s0: float; spot
    :param k: float; strike
    :param t: float; maturity
    :param r: float; short term interest rate
    :param sigma: volatility factor
    :return: partial derivative of BSM w.r.t. sigma (i.e. Vega)
    """
    from math import log, sqrt
    from scipy import stats

    s0 = float(s0)
    d1 = (log(s0/k) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    vega = s0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(t)
    return vega


# Implied Volatility Function
def bsm_call_imp_vol(s0, k, t, r, c0, sigma_est, it=100):
    """
    Implied volatility of European Call option in BSM model
    :param s0: float; spot
    :param k: float; strike
    :param t: float; maturity
    :param r: float; risk free rate
    :param c0: float; call option price?
    :param sigma_est: float; estimate of implied volatility
    :param it: integer; number of iterations
    :return: sigma_est; float of estimated implied volatility
    """
    for i in range(it):
        sigma_est -= ((bsm_call_value(s0, k, t, r, sigma_est) - c0)/(bsm_vega(s0, k, t, r, sigma_est)))

    return sigma_est


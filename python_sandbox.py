from math import exp, log, sqrt

from scipy.stats import norm


def call_option_pricer(spot, strike, maturity, r, vol):
    """Return the Black-Scholes price for a European call option."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0 or vol <= 0:
        raise ValueError("maturity and vol must be positive")

    d1 = (log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)
    price = spot * norm.cdf(d1) - strike * exp(-r * maturity) * norm.cdf(d2)
    return price
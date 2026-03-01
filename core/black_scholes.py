"""Black-Scholes pricing + Newton-Raphson Implied Volatility (single shared copy)."""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> float:
    """Newton-Raphson IV. Returns decimal IV (e.g. 0.20 = 20%)."""
    if T <= 0 or price <= 0:
        return 0.0
    sigma = 0.5
    for _ in range(max_iter):
        p = black_scholes_call(S, K, T, r, sigma) if option_type == "CE" else black_scholes_put(S, K, T, r, sigma)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega < tol:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001
            break
    return max(0.0, sigma)

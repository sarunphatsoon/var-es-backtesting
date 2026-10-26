import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_1samp
from scipy.special import xlogy

def hits(returns, var):
    r = returns.loc[var.index]
    return (r < -var).astype(int)

def _ll(n_fail, n_ok, p):
    return xlogy(n_ok, 1 - p) + xlogy(n_fail, p)

def kupiec(h, alpha):
    n, x = len(h), h.sum()
    lr = -2 * (_ll(x, n - x, alpha) - _ll(x, n - x, x / n))
    return lr, chi2.sf(lr, 1)

def christoffersen_ind(h):
    h = h.values
    prev, curr = h[:-1], h[1:]
    n00 = ((prev == 0) & (curr == 0)).sum()
    n01 = ((prev == 0) & (curr == 1)).sum()
    n10 = ((prev == 1) & (curr == 0)).sum()
    n11 = ((prev == 1) & (curr == 1)).sum()
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _ll(n01 + n11, n00 + n10, pi)
    ll1 = _ll(n01, n00, pi01) + _ll(n11, n10, pi11)
    lr = -2 * (ll0 - ll1)
    return lr, chi2.sf(lr,1)

def conditional_coverage(h, alpha):
    lr = kupiec(h, alpha)[0] + christoffersen_ind(h)[0]
    return lr, chi2.sf(lr, 2)

def es_ratio_test(returns, var, es):
    r = returns.loc[var.index]
    m = r < -var
    if m.sum() < 5:
        return np.nan, np.nan
    ratio = (-r[m]) / es[m]
    return ratio.mean(), ttest_1samp(ratio, 1.0).pvalue

def traffic_light(h, window = 250):
    x = h.iloc[-window:].sum()
    return "green" if x<=4 else "yellow" if x <= 9 else "red"

def fz0(returns, var, es, alpha):
    r = returns.loc[var.index]
    v, e = -var, -es
    s = -(1 / (alpha * e)) * (r <= v) * (v - r) + v / e + np.log(-e) - 1
    return s.mean()

def summarize(returns, fc, alpha):
    h = hits(returns, fc["VaR"])
    return {
        "n" : len(h),
        "hits" : int(h.sum()),
        "expected" : round(alpha * len(h), 1),
        "hit_rate" : h.mean(),
        "kupiec_p" : kupiec(h, alpha)[1],
        "indep_p" : christoffersen_ind(h)[1],
        "cc_p" : conditional_coverage(h, alpha)[1],
        "es_ratio" : es_ratio_test(returns, fc["VaR"], fc["ES"])[0],
        "fz0" : fz0(returns, fc["VaR"], fc["ES"], alpha),
    }

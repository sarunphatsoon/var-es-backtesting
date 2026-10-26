import numpy as np #arrays
import pandas as pd #tables and time series
from scipy.stats import norm, t #normal and t-distributions
from arch import arch_model #GARCH fitting

def to_returns(prices):
    return np.log(prices.dropna()).diff().dropna()


def historical(returns, window = 250, alpha = 0.05):
    rows = []
    for i in range(window, len(returns)):
        past = returns.iloc[i - window:i]
        q = np.quantile(past, alpha)
        tail = past[past <= q]
        rows.append((returns.index[i], -q, -tail.mean()))
    return pd.DataFrame(rows, columns=["Date", "VaR", "ES"]).set_index("Date")



def normal(returns, window = 250, alpha = 0.05):
    mu = returns.rolling(window).mean().shift(1)
    sd = returns.rolling(window).std().shift(1)
    z = norm.ppf(alpha)
    var = -(mu + sd * z)
    es = -mu + sd * norm.pdf(z) / alpha
    return pd.DataFrame({"VaR":var, "ES":es}).dropna()


def ewma(returns, window = 250, alpha = 0.05, lam = 0.94):
    v = returns.pow(2).ewm(alpha = 1 - lam, adjust = False).mean().shift(1)
    sd = np.sqrt(v)
    z = norm.ppf(alpha)
    out = pd.DataFrame({"VaR": -sd *z,
                         "ES": sd * norm.pdf(z) / alpha})
    return out.iloc[window:]




def student_t(returns, window = 500, alpha = 0.05, step = 20):
    rows = []
    for start in range(window, len(returns), step):
        nu, loc, scale = t.fit(returns.iloc[start - window:start])
        tq = t.ppf(alpha, nu)
        var = -(loc + scale * tq)
        es = -loc + scale * t.pdf(tq, nu) / alpha * (nu + tq**2) / (nu - 1)
        for i in range(start, min(start + step, len(returns))):
            rows.append((returns.index[i], var, es))
    return pd.DataFrame(rows, columns = ["Date", "VaR", "ES"]).set_index("Date")




def garch(returns, window = 500, alpha = 0.05, step = 20, method = "t" ):
    r = returns * 100
    rows = []
    for start in range(window, len(r), step):
        res = arch_model(r.iloc[start - window:start], mean = "Constant", vol = "GARCH", p = 1, q = 1, dist = "t").fit(disp = "off")
        mu, omega = res.params['mu'], res.params["omega"]
        a, b, nu = res.params['alpha[1]'], res.params['beta[1]'], res.params["nu"]
        sigma2 = res.forecast(horizon = 1).variance.iloc[-1,0]
        z = res.std_resid.dropna() 
        for i in range(start, min(start + step, len(r))):
            sigma = np.sqrt(sigma2)
            if method == "t":
                k = np.sqrt((nu - 2)/ nu)
                tq = t.ppf(alpha, nu)
                var = -(mu + sigma * k * tq)
                es = -mu + sigma * k * t.pdf(tq, nu) / alpha * (nu + tq**2) / (nu-1)
            else:
                qz = np.quantile(z, alpha)
                var = -(mu + sigma * qz)
                es = -(mu + sigma * z[z <= qz].mean())
            rows.append((r.index[i], var / 100, es/100))
            eps = r.iloc[i] - mu
            sigma2 = omega + a * eps**2 + b * sigma2
    return pd.DataFrame(rows, columns = ["Date", "VaR", "ES"]).set_index("Date")


if __name__ == "__main__":
    df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates = ["Date"], index_col = "Date")
    df = df.iloc[:-1]
    
    r = to_returns(df["SPY"]) #SPY first
    alpha = 0.01
    
    methods = {
        "historical" : historical(r, alpha = alpha),
        "normal" : normal(r, alpha = alpha),
        "ewma" : ewma(r, alpha = alpha),
        "student_t" : student_t(r, alpha = alpha),
        "garch_t" : garch(r, alpha = alpha, method = "t"),
        "garch_fhs" : garch(r, alpha = alpha, method = "fhs"),
        }
    
    for name, fc in methods.items():
        actual = r.loc[fc.index]
        hit_rate = (actual < -fc["VaR"]).mean()
        print(f"{name:12s} hit rate = {hit_rate:.3f} (target = {alpha}) avg VaR = {fc["VaR"].mean():.4f}")
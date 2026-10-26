import pickle
import time
import pandas as pd

from models import to_returns, historical, normal, ewma, student_t, garch
from backtest import hits, summarize, traffic_light

DATA_FILE = "portfolio_risk_dataset.csv"
ASSETS = ["SPY", "KRE", "JPM", "AAPL", "TSLA", "ES3.SI", "BTC-USD"]
ALPHAS = [0.05, 0.01]

METHODS = {
    "historical" : lambda r, a: historical(r, alpha=a),
    "normal" : lambda r, a: normal(r, alpha = a),
    "ewma" : lambda r, a: ewma(r, alpha = a),
    "student_t" : lambda r, a: student_t(r, alpha = a),
    "garch_t" : lambda r, a: garch(r, alpha = a, method = "t"),
    "garch_fhs" : lambda r, a: garch(r, alpha= a, method = "fhs"),
}

def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    df = df.iloc[:-1]

    rows = []
    forecasts = {}
    t0 = time.time()

    for asset in ASSETS:
        r = to_returns(df[asset])
        for alpha in ALPHAS:
            fcs = {name: fn(r, alpha) for name, fn in METHODS.items()}
            start = max(fc.index[0] for fc in fcs.values())
            fcs = {name: fc.loc[start:] for name, fc in fcs.items()}

            for name, fc in fcs.items():
                row = {'asset' : asset, "alpha" : alpha, "method" : name}
                row.update(summarize(r, fc, alpha))
                if alpha == 0.01:
                    row["basel_last250"] = traffic_light(hits(r, fc["VaR"]))
                rows.append(row)
                forecasts[(asset, alpha, name)] = fc

            print(f"{asset:8s} alpha = {alpha} done ({time.time() - t0:.0f}s)", flush = True)
    results = pd.DataFrame(rows)
    results.to_csv("results.csv", index=False)
    with open("forecast.pkl", "wb") as f:
        pickle.dump(forecasts, f)

    show = ["asset", "alpha", "method", "n", "hits", "expected", "kupiec_p", "indep_p", "cc_p", "es_ratio", "fz0"]
    print(results[show].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
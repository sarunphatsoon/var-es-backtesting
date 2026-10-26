import pickle
import pandas as pd
import matplotlib.pyplot as plt
from models import to_returns

METHODS = ["historical", "normal", "ewma", "student_t", "garch_t", "garch_fhs"]
ASSETS = ["SPY", "KRE", "JPM", "AAPL", "TSLA", "ES3.SI", "BTC-USD"]

with open("forecast.pkl", "rb") as f:
    forecast = pickle.load(f)

df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates=["Date"], index_col= "Date")
df = df.iloc[:-1]
r = to_returns(df["KRE"])

ASSET, ALPHA = "KRE", 0.01
START, END = "2023-02-20", "2023-04-14"

var = pd.DataFrame({m: forecast[(ASSET, ALPHA, m)]["VaR"] for m in METHODS}).loc[START:END]
loss = (-r).loc[START:END]

SLOW = ["historical", "normal", "student_t"]
fig, ax = plt.subplots(figsize = (11,6))
ax.bar(loss.index, loss * 100, width=0.8, color = "lightgray", label = "KRE daily loss")
for m in METHODS:
    ax.plot(var.index, var[m] * 100, label = m, linewidth = 1.8, linestyle = "--" if m in SLOW else "-")
ax.axhline(0, color = "black", linewidth = 0.6)
ax.set_title("KRE : 99% VaR forecasts vs actual losses, Feb-Apr 2023")
ax.set_ylabel("Loss / VaR (% of value); negative bars = gains")
ax.legend(loc = "upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("kre_march2023.png", dpi=500)

res = pd.read_csv("results.csv")
res99 = res[res["alpha"] == 0.01].copy()
res99["ratio"] = res99["hits"] / res99["expected"]
table = res99.pivot(index = "asset", columns="method", values = "ratio").loc[ASSETS, METHODS]

fig2, ax2 = plt.subplots(figsize = (11,6))
table.plot(kind = "bar", ax = ax2, width = 0.85)
ax2.axhline(1.0, color = "black", linestyle = "--", linewidth = 1)
ax2.set_title("99% VaR: exceedances / expected (1.0 = perfectly calibrated)")
ax2.set_ylabel("exceedances / expected")
ax2.set_xlabel("")
ax2.tick_params(axis = "x", rotation = 0)
fig2.tight_layout()
fig2.savefig("calibration_99.png", dpi=150)
plt.show()
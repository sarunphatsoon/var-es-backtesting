import pickle
import pandas as pd
from models import to_returns

with open("forecast.pkl", "rb") as f:
    forecasts = pickle.load(f)

df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates=["Date"], index_col="Date")
df = df.iloc[:-1]
r = to_returns(df["KRE"])

ASSET, ALPHA = "KRE", 0.01
METHODS = ["historical", "normal", "ewma", "student_t", "garch_t", "garch_fhs"]
START, END = "2023-03-06", "2023-03-17"

var = pd.DataFrame({m: forecasts[(ASSET, ALPHA, m)]["VaR"] for m in METHODS}).loc[START:END]
ret = r.loc[START:END]
breach = (-var).gt(ret, axis = 0)

print("Actual KRE returns (%)")
print((ret * 100).round(2))
print("\n99% VaR forecasts (%) -- the loss each model said would be exceeded 1% of the time")
print((var * 100).round(2))
print("\nBreaches (True = loss beat the VaR)")
print(breach)
print("\nTotal breaches per method:")
print(breach.sum())
print("\nHow much each VaR grew from Mar 6 to Mar 17 (ratio)")
print((var.loc["2023-03-17"] / var.loc["2023-03-06"]).round(2))
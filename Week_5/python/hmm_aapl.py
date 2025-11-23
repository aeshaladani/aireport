# hmm_aapl.py
# Python3 script to download AAPL, compute returns, fit Gaussian HMM
# Required: pip install yfinance numpy pandas matplotlib hmmlearn

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# ---------- download data ----------
ticker = "AAPL"
data = yf.download(ticker, start="2014-01-01", end="2024-01-01")
data['Return'] = data['Adj Close'].pct_change().fillna(0)

returns = data['Return'].values.reshape(-1,1)

# save returns plot
plt.figure(figsize=(10,4))
plt.plot(data.index, data['Return'])
plt.title("AAPL daily returns (2014-2023)")
plt.savefig("figures/fig_aapl_returns.png", dpi=150)
plt.close()

# ---------- fit 2-state HMM ----------
model2 = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=0)
model2.fit(returns)
states2 = model2.predict(returns)

# Plot returns colored by inferred state
plt.figure(figsize=(10,4))
for s in np.unique(states2):
    mask = (states2 == s)
    plt.plot(data.index[mask], data['Return'][mask], '.', label=f"state {s}", alpha=0.6)
plt.legend()
plt.title("AAPL returns colored by inferred 2-state HMM")
plt.savefig("figures/fig_hmm_states_2.png", dpi=150)
plt.close()

print("2-state HMM means:", model2.means_.ravel())
print("2-state HMM covars:", model2.covars_.ravel())
print("Transition matrix:\n", model2.transmat_)

# ---------- fit 3-state HMM ----------
model3 = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=1)
model3.fit(returns)
states3 = model3.predict(returns)

# plot 3-state
plt.figure(figsize=(10,4))
for s in np.unique(states3):
    mask = (states3 == s)
    plt.plot(data.index[mask], data['Return'][mask], '.', label=f"state {s}", alpha=0.6)
plt.legend()
plt.title("AAPL returns colored by inferred 3-state HMM")
plt.savefig("figures/fig_hmm_states_3.png", dpi=150)
plt.close()

print("3-state HMM means:", model3.means_.ravel())
print("3-state HMM covars:", model3.covars_.ravel())
print("Transition matrix:\n", model3.transmat_)

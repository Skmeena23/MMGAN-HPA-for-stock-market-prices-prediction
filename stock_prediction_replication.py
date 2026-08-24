"""
Replicating MMGAN-HPA (Polamuri et al., 2022) — Hybrid Stock Price Prediction
==============================================================================

This script:
  1. Loads daily OHLCV data for six NSE-listed stocks (TCS, BHEL, WIPRO,
     AXISBANK, MARUTI, TATASTEEL), matching the tickers used in the paper.
  2. Runs exploratory data analysis (rolling averages, return-correlation
     heatmap, volume-vs-volatility scatter).
  3. Builds a hybrid prediction model per ticker: Linear Regression (linear
     component) + Random Forest Regressor (non-linear component), averaged
     together — echoing the paper's MM-HPA / MMGAN-HPA linear+non-linear
     fusion idea, without the full LSTM-generator/CNN-discriminator GAN.
  4. Evaluates each ticker with MAE, MSE, and correlation between actual
     and predicted closing prices on a held-out chronological test split.

Data expected at: ./data/<TICKER>.csv with columns
  Date, Close, High, Low, Open, Volume  (first two header rows after the
  real header are skipped — adjust `skiprows` in `load_ticker` if your
  CSV export doesn't have that quirk).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TICKERS = ["TCS", "BHEL", "WIPRO", "AXISBANK", "MARUTI", "TATASTEEL"]
DATA_DIR = "data"
CHART_DIR = "charts"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load_ticker(ticker: str) -> pd.DataFrame:
    """Load and clean one ticker's CSV file."""
    df = pd.read_csv(
        f"{DATA_DIR}/{ticker}.csv",
        skiprows=[1, 2],  # drop non-data header rows some exports include
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        header=0,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def make_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Build lagged-price, moving-average, and volatility features.

    All features are shifted so that only past information is used to
    predict the current day's close (no look-ahead leakage).
    """
    d = df.copy()
    for lag in range(1, lags + 1):
        d[f"lag_{lag}"] = d["Close"].shift(lag)
    d["ma_5"] = d["Close"].rolling(5).mean().shift(1)
    d["ma_10"] = d["Close"].rolling(10).mean().shift(1)
    d["vol_5"] = d["Close"].pct_change().rolling(5).std().shift(1)
    return d.dropna().reset_index(drop=True)


def run_eda(data: dict):
    """Generate the three EDA charts used in the slide deck."""
    # 1. Rolling average (TCS as the lead example)
    df = data["TCS"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="#1f77b4", linewidth=1, alpha=0.6)
    ax.plot(df["Date"], df["Close"].rolling(30).mean(), label="30-day Rolling Avg", color="#d62728", linewidth=1.8)
    ax.plot(df["Date"], df["Close"].rolling(100).mean(), label="100-day Rolling Avg", color="#2ca02c", linewidth=1.8)
    ax.set_title("TCS Closing Price with Rolling Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/rolling_avg_tcs.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap of daily returns across all tickers
    returns = pd.DataFrame({t: data[t].set_index("Date")["Close"].pct_change() for t in TICKERS})
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation of Daily Returns Across Stocks")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    # 3. Scatter: trading volume vs. absolute daily return (TCS)
    df = data["TCS"].copy()
    df["AbsReturn"] = df["Close"].pct_change().abs()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["Volume"], df["AbsReturn"] * 100, alpha=0.4, s=18, color="#9467bd")
    ax.set_xlabel("Trading Volume")
    ax.set_ylabel("Absolute Daily Return (%)")
    ax.set_title("TCS: Trading Volume vs. Price Volatility")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/scatter_volume_volatility.png", dpi=150)
    plt.close()

    return corr


def train_hybrid_model(df: pd.DataFrame, train_frac: float = 0.85):
    """Train the linear + non-linear hybrid model for one ticker.

    Returns a dict with metrics and the test-set predictions/dates for plotting.
    """
    feat = make_features(df)
    feature_cols = [c for c in feat.columns if c.startswith(("lag_", "ma_", "vol_"))]

    X = feat[feature_cols].values
    y = feat["Close"].values
    dates = feat["Date"].values

    split = int(len(feat) * train_frac)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    linear_model = LinearRegression().fit(X_train, y_train)
    pred_linear = linear_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42).fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

    # Hybrid = simple average of linear + non-linear predictions
    pred_hybrid = (pred_linear + pred_rf) / 2

    mae = mean_absolute_error(y_test, pred_hybrid)
    mse = mean_squared_error(y_test, pred_hybrid)
    corr = np.corrcoef(y_test, pred_hybrid)[0, 1]

    return {
        "MAE": mae,
        "MSE": mse,
        "Correlation": corr,
        "dates_test": dates_test,
        "y_test": y_test,
        "pred_hybrid": pred_hybrid,
    }


def plot_predictions(results: dict):
    """Grid of actual-vs-predicted plots, one panel per ticker."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, ticker in zip(axes.flat, TICKERS):
        r = results[ticker]
        ax.plot(r["dates_test"], r["y_test"], label="Actual", color="#d62728", linewidth=1.3)
        ax.plot(r["dates_test"], r["pred_hybrid"], label="Predicted (Hybrid)",
                color="#1f77b4", linewidth=1.3, linestyle="--")
        ax.set_title(ticker)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Hybrid Model (Linear + Random Forest): Actual vs. Predicted Close Price — Test Set")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{CHART_DIR}/hybrid_predictions_grid.png", dpi=150)
    plt.close()


def main():
    import os
    os.makedirs(CHART_DIR, exist_ok=True)

    data = {t: load_ticker(t) for t in TICKERS}
    print("Loaded tickers:", list(data.keys()))

    corr = run_eda(data)
    print("\nReturn correlation matrix:\n", corr.round(3))

    results = {t: train_hybrid_model(data[t]) for t in TICKERS}

    summary = pd.DataFrame([
        {"Ticker": t, "MAE": r["MAE"], "MSE": r["MSE"], "Correlation": r["Correlation"]}
        for t, r in results.items()
    ])
    print("\nHybrid model performance:\n", summary.to_string(index=False))
    summary.to_csv("model_results.csv", index=False)

    plot_predictions(results)
    print(f"\nCharts saved to ./{CHART_DIR}/, metrics saved to model_results.csv")


if __name__ == "__main__":
    main()

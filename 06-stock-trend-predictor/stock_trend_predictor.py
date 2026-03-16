"""
Stock Trend Predictor
=====================
A beginner-friendly demonstration of time-series analysis using synthetic stock data.

This project covers:
- Generating realistic synthetic stock price data
- Computing Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
- Identifying buy/sell signals using moving-average crossover strategy
- Predicting short-term trend direction with a basic ML classifier
- Visualising results with matplotlib

No real financial data or API keys are required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────

def generate_stock_data(
    start: str = "2022-01-01",
    periods: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily stock price data that mimics a real price series.

    Uses a Geometric Brownian Motion (GBM) model — the same mathematical
    foundation used by the Black-Scholes options pricing formula.

    Parameters
    ----------
    start   : Start date string (YYYY-MM-DD)
    periods : Number of trading days to simulate
    seed    : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start=start, periods=periods)  # business days only

    # GBM parameters
    mu = 0.0003       # daily drift (small upward trend)
    sigma = 0.015     # daily volatility

    # Simulate log returns and convert to price series
    log_returns = np.random.normal(mu, sigma, periods)
    prices = 100.0 * np.exp(np.cumsum(log_returns))  # start at $100

    # Build OHLCV from close prices
    noise = np.random.uniform(0.002, 0.008, periods)  # intraday spread
    df = pd.DataFrame({
        "date":   dates,
        "close":  prices,
        "open":   prices * (1 + np.random.uniform(-0.003, 0.003, periods)),
        "high":   prices * (1 + noise),
        "low":    prices * (1 - noise),
        "volume": np.random.randint(500_000, 5_000_000, periods),
    })
    df.set_index("date", inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Add Simple Moving Average (SMA) and Exponential Moving Average (EMA) columns.

    Parameters
    ----------
    df    : DataFrame with a 'close' column
    short : Window size for the short-term average (default 20 days)
    long  : Window size for the long-term average (default 50 days)

    Returns
    -------
    DataFrame with added columns: sma_short, sma_long, ema_short, ema_long
    """
    df = df.copy()
    df[f"sma_{short}"] = df["close"].rolling(window=short).mean()
    df[f"sma_{long}"]  = df["close"].rolling(window=long).mean()
    df[f"ema_{short}"] = df["close"].ewm(span=short, adjust=False).mean()
    df[f"ema_{long}"]  = df["close"].ewm(span=long,  adjust=False).mean()
    return df


def add_features(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Engineer features used by the ML model.

    Features:
    - Normalised price distance from each moving average
    - Rate of change (momentum) over 5 and 10 days
    - Rolling volatility (std of returns over 10 days)

    Parameters
    ----------
    df : DataFrame with OHLCV data and moving average columns

    Returns
    -------
    DataFrame with additional feature columns and a binary 'target' column
    (1 = price higher in 5 days, 0 = lower)
    """
    df = df.copy()

    # Distance from moving averages (normalised)
    df["dist_sma_short"] = (df["close"] - df[f"sma_{short}"]) / df[f"sma_{short}"]
    df["dist_sma_long"]  = (df["close"] - df[f"sma_{long}"])  / df[f"sma_{long}"]

    # Momentum: percentage change over N days
    df["mom_5"]  = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)

    # Rolling volatility (standard deviation of daily returns)
    daily_ret = df["close"].pct_change()
    df["volatility_10"] = daily_ret.rolling(10).std()

    # SMA crossover ratio (positive = short above long = bullish)
    df["sma_cross_ratio"] = (df[f"sma_{short}"] - df[f"sma_{long}"]) / df[f"sma_{long}"]

    # Binary label: 1 if close is higher 5 days from now
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)

    return df


# ─────────────────────────────────────────────
# 3. TRADING SIGNALS
# ─────────────────────────────────────────────

def add_crossover_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Generate buy / sell signals based on simple moving average crossover.

    Strategy:
    - BUY  signal (+1): short SMA crosses ABOVE long SMA
    - SELL signal (-1): short SMA crosses BELOW long SMA

    Parameters
    ----------
    df : DataFrame with sma_short and sma_long columns

    Returns
    -------
    DataFrame with a 'signal' column (1 = buy, -1 = sell, 0 = hold)
    """
    df = df.copy()
    prev_short = df[f"sma_{short}"].shift(1)
    prev_long  = df[f"sma_{long}"].shift(1)

    df["signal"] = 0
    # Buy: short crosses above long
    df.loc[(df[f"sma_{short}"] > df[f"sma_{long}"]) & (prev_short <= prev_long), "signal"] = 1
    # Sell: short crosses below long
    df.loc[(df[f"sma_{short}"] < df[f"sma_{long}"]) & (prev_short >= prev_long), "signal"] = -1

    return df


# ─────────────────────────────────────────────
# 4. ML MODEL
# ─────────────────────────────────────────────

FEATURE_COLS = ["dist_sma_short", "dist_sma_long", "mom_5", "mom_10",
                "volatility_10", "sma_cross_ratio"]


def train_model(df: pd.DataFrame):
    """
    Train a Gradient Boosting classifier to predict 5-day price direction.

    Parameters
    ----------
    df : DataFrame with feature and target columns

    Returns
    -------
    Tuple of (trained model, scaler, test metrics dict)
    """
    # Drop rows with NaN values introduced by rolling windows / shifts
    clean = df[FEATURE_COLS + ["target"]].dropna()

    X = clean[FEATURE_COLS].values
    y = clean["target"].values

    # Train / test split — preserve time order (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report":   classification_report(y_test, y_pred, target_names=["Down", "Up"]),
    }
    return model, scaler, metrics


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(df: pd.DataFrame, output_path: str = "stock_trend_predictor.png") -> None:
    """
    Plot stock price with moving averages, volume, and buy/sell signals.

    Parameters
    ----------
    df          : Processed DataFrame (must contain moving average and signal columns)
    output_path : File path where the chart PNG will be saved
    """
    # Show only the last 200 rows for clarity
    plot_df = df.dropna(subset=["sma_20", "sma_50"]).tail(200)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle("Stock Trend Predictor – SMA Crossover Strategy", fontsize=14, fontweight="bold")

    # ── Price & moving averages ──────────────────────────────────────────
    ax1.plot(plot_df.index, plot_df["close"],   color="steelblue",  lw=1.5, label="Close Price")
    ax1.plot(plot_df.index, plot_df["sma_20"],  color="orange",     lw=1.2, linestyle="--", label="SMA 20")
    ax1.plot(plot_df.index, plot_df["sma_50"],  color="purple",     lw=1.2, linestyle="--", label="SMA 50")
    ax1.plot(plot_df.index, plot_df["ema_20"],  color="green",      lw=0.8, linestyle=":",  label="EMA 20", alpha=0.7)

    # Buy / sell markers
    buy_signals  = plot_df[plot_df["signal"] ==  1]
    sell_signals = plot_df[plot_df["signal"] == -1]
    ax1.scatter(buy_signals.index,  buy_signals["close"],  marker="^", color="limegreen", s=120, zorder=5, label="Buy Signal")
    ax1.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red",       s=120, zorder=5, label="Sell Signal")

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Volume bar chart ─────────────────────────────────────────────────
    colors = ["limegreen" if r >= o else "red"
              for r, o in zip(plot_df["close"], plot_df["open"])]
    ax2.bar(plot_df.index, plot_df["volume"] / 1_000_000, color=colors, alpha=0.7)
    ax2.set_ylabel("Volume (M)")
    ax2.grid(alpha=0.3)

    # Date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Stock Trend Predictor")
    print("=" * 55)

    # Step 1 – Generate data
    print("\n[1] Generating synthetic stock data (500 trading days)…")
    df = generate_stock_data(periods=500)
    print(f"    Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"    Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f}")

    # Step 2 – Add technical indicators
    print("\n[2] Computing moving averages & features…")
    df = add_moving_averages(df, short=20, long=50)
    df = add_features(df, short=20, long=50)
    df = add_crossover_signals(df, short=20, long=50)

    # Step 3 – Count signals
    buy_count  = (df["signal"] ==  1).sum()
    sell_count = (df["signal"] == -1).sum()
    print(f"    Buy signals  : {buy_count}")
    print(f"    Sell signals : {sell_count}")

    # Step 4 – Train ML model
    print("\n[3] Training Gradient Boosting classifier…")
    model, scaler, metrics = train_model(df)
    print(f"    Test accuracy : {metrics['accuracy']:.2%}")
    print("\n    Classification Report:")
    for line in metrics["report"].strip().split("\n"):
        print(f"    {line}")

    # Step 5 – Feature importance
    print("\n[4] Feature importances:")
    importance_pairs = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for feat, imp in importance_pairs:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<22} {bar} {imp:.3f}")

    # Step 6 – Plot
    print("\n[5] Generating chart…")
    plot_results(df)

    # Step 7 – Quick prediction on latest data
    print("\n[6] Predicting trend for the next 5 days…")
    latest = df[FEATURE_COLS].dropna().tail(1)
    latest_scaled = scaler.transform(latest)
    prediction = model.predict(latest_scaled)[0]
    probability = model.predict_proba(latest_scaled)[0]
    trend = "📈 UP" if prediction == 1 else "📉 DOWN"
    confidence = max(probability)
    print(f"    Last close : ${df['close'].iloc[-1]:.2f}")
    print(f"    Prediction : {trend}  (confidence: {confidence:.1%})")

    print("\n✅ Done! Chart saved as stock_trend_predictor.png")
    print("=" * 55)

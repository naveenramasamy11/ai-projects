# 📈 Project 06 – Stock Trend Predictor

> Predict short-term stock price direction using moving averages, crossover signals, and a Gradient Boosting classifier — no API keys required.

---

## 📁 File Table

| File | Purpose |
|------|---------|
| `stock_trend_predictor.py` | Main Python script — runs the full pipeline end-to-end |
| `stock_trend_predictor.ipynb` | Step-by-step Jupyter notebook with explanations |
| `requirements.txt` | Python dependency list with pinned versions |
| `stock_trend_predictor.png` | Chart output (generated on first run) |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python stock_trend_predictor.py
```

Or open the notebook:

```bash
jupyter notebook stock_trend_predictor.ipynb
```

---

## 📊 Example Output

```
=======================================================
  Stock Trend Predictor
=======================================================

[1] Generating synthetic stock data (500 trading days)…
    Date range : 2022-01-03 → 2023-11-17
    Price range: $82.34 – $141.56

[2] Computing moving averages & features…
    Buy signals  : 7
    Sell signals : 7

[3] Training Gradient Boosting classifier…
    Test accuracy : 57.00%

    Classification Report:
                  precision  recall  f1-score  support
    Down             0.40      0.44      0.42       41
    Up               0.66      0.63      0.64       59

[4] Feature importances:
    mom_10                 ████████████████████ 0.212
    mom_5                  ███████████████████  0.204
    sma_cross_ratio        ████████████████     0.178
    dist_sma_short         ███████████████      0.163
    volatility_10          ████████████         0.132
    dist_sma_long          ██████████           0.111

[6] Predicting trend for the next 5 days…
    Last close : $128.74
    Prediction : 📈 UP  (confidence: 63.4%)

✅ Done! Chart saved as stock_trend_predictor.png
=======================================================
```

---

## 🧠 Key Concepts

| Concept | Plain English Explanation |
|---------|--------------------------|
| **Simple Moving Average (SMA)** | Average of the last N closing prices — smooths out noise |
| **Exponential Moving Average (EMA)** | Weighted average where recent prices matter more |
| **SMA Crossover** | Buy/sell signal when short SMA crosses over long SMA |
| **Momentum** | How much price changed over a fixed number of days |
| **Volatility** | Standard deviation of daily returns — measures price jumpiness |
| **Gradient Boosting** | Ensemble of small decision trees, each correcting the last |
| **Feature Engineering** | Converting raw prices into ML-friendly numeric inputs |
| **Geometric Brownian Motion** | Mathematical model used to simulate realistic stock prices |

---

## 📦 Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| `numpy` | Numerical computations, random simulation | [numpy.org](https://numpy.org) |
| `pandas` | Time-indexed DataFrames, rolling statistics | [pandas.pydata.org](https://pandas.pydata.org) |
| `matplotlib` | Multi-panel charts, price/volume plots | [matplotlib.org](https://matplotlib.org) |
| `scikit-learn` | Gradient Boosting, scaling, evaluation | [scikit-learn.org](https://scikit-learn.org) |

---

## 💡 Difficulty: ⭐⭐ Intermediate

Recommended after completing Projects 01–05. Builds on classification concepts and introduces time-series thinking.

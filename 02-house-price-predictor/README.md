# 🏠 Project 02: House Price Predictor

Predict house prices using **Linear Regression** — a foundational supervised machine learning algorithm.

## 🧠 What It Does

Given a house's features, the model predicts its price:

| Input Feature | Example |
|--------------|---------|
| Size (sq ft) | 2,500 |
| Bedrooms | 4 |
| Age (years) | 10 |
| Distance from city (km) | 8.5 |
| **→ Predicted Price** | **$238,000** |

## 📁 Files

| File | Description |
|------|-------------|
| `house_price_predictor.py` | Ready-to-run Python script |
| `house_price_predictor.ipynb` | Step-by-step Jupyter Notebook (recommended for learning) |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python house_price_predictor.py
```

Or open the notebook (recommended):
```bash
jupyter notebook house_price_predictor.ipynb
```

## 📊 What the Script Outputs

```
MODEL PERFORMANCE (on unseen test data)
  Mean Absolute Error (MAE) :      $19,247
  Root Mean Squared Error   :      $24,631
  R² Score                  :       0.9721  (1.0 = perfect)

FEATURE IMPORTANCE
  size_sqft      :     +89,412   ↑ raises price
  bedrooms       :     +27,331   ↑ raises price
  distance_km    :     -18,204   ↓ lowers price
  age_years      :      -9,876   ↓ lowers price
```

## 🔑 Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **Supervised learning** | Model learns from labeled examples (features + correct answers) |
| **Linear Regression** | Finds the best-fit line/surface through the data |
| **Train/Test split** | Train on 80%, test on hidden 20% to measure real performance |
| **R² Score** | How well the model explains the data (0 = bad, 1 = perfect) |
| **MAE** | Average dollar amount your predictions are off by |

## 🛠 Libraries Used

- [`scikit-learn`](https://scikit-learn.org/) — LinearRegression, train/test split, metrics
- [`pandas`](https://pandas.pydata.org/) — Data manipulation
- [`numpy`](https://numpy.org/) — Numerical computing
- [`matplotlib`](https://matplotlib.org/) — Visualization

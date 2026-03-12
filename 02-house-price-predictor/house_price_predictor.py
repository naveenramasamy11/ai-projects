"""
=============================================================
  Project 02: House Price Predictor
  ------------------------------------
  This script uses Linear Regression to predict house prices
  based on features like size, number of bedrooms, and age.

  It demonstrates a complete ML workflow:
    1. Generate / load data
    2. Explore and visualize the data
    3. Split into training and test sets
    4. Train a Linear Regression model
    5. Evaluate the model
    6. Make predictions on new houses
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)


# ── 1. Generate a realistic synthetic dataset ────────────────

def generate_dataset(n_samples: int = 300) -> pd.DataFrame:
    """
    Create a synthetic housing dataset.

    Features:
      - size_sqft     : house size in square feet (500–4000)
      - bedrooms      : number of bedrooms (1–6)
      - age_years     : house age in years (0–50)
      - distance_km   : distance from city center in km (1–30)

    Price formula (with some random noise):
      price ≈ 50 * size_sqft + 10,000 * bedrooms
              - 2,000 * age_years - 5,000 * distance_km
              + base_price + noise
    """
    size_sqft = np.random.randint(500, 4001, n_samples).astype(float)
    bedrooms = np.random.randint(1, 7, n_samples).astype(float)
    age_years = np.random.randint(0, 51, n_samples).astype(float)
    distance_km = np.random.uniform(1, 30, n_samples)

    noise = np.random.normal(0, 20_000, n_samples)
    price = (
        50 * size_sqft
        + 10_000 * bedrooms
        - 2_000 * age_years
        - 5_000 * distance_km
        + 80_000
        + noise
    ).clip(min=50_000)  # no house cheaper than $50k

    return pd.DataFrame({
        "size_sqft": size_sqft,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "distance_km": distance_km.round(1),
        "price": price.round(-3),  # round to nearest $1,000
    })


# ── 2. Train and evaluate the model ─────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train a Linear Regression model and print evaluation metrics.
    Returns the trained model and scaler.
    """
    features = ["size_sqft", "bedrooms", "age_years", "distance_km"]
    X = df[features]
    y = df["price"]

    # Split: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (good practice, even for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # ── Evaluation metrics ──────────────────────────────────
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("  MODEL PERFORMANCE (on unseen test data)")
    print("=" * 55)
    print(f"  Mean Absolute Error (MAE) : ${mae:>12,.0f}")
    print(f"  Root Mean Squared Error   : ${rmse:>12,.0f}")
    print(f"  R² Score                  : {r2:>13.4f}  (1.0 = perfect)")
    print()
    print("  ℹ️  MAE = on average the model is off by this much")
    print("  ℹ️  R² tells how well the model explains price variation")
    print("     0.90+ is great, 0.70–0.90 is decent, <0.70 is weak")
    print("=" * 55)

    # ── Feature importance ──────────────────────────────────
    print("\n  FEATURE IMPORTANCE (Linear Coefficients)")
    print("-" * 40)
    for feat, coef in zip(features, model.coef_):
        direction = "↑ raises price" if coef > 0 else "↓ lowers price"
        print(f"  {feat:<15}: {coef:>+12,.0f}   {direction}")

    return model, scaler, X_test, y_test, y_pred


# ── 3. Visualize results ─────────────────────────────────────

def plot_results(y_test, y_pred, df):
    """Create 3 charts: actual vs predicted, residuals, price vs size."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("House Price Predictor — Results", fontsize=14, fontweight="bold")

    # --- Chart 1: Actual vs Predicted ---
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, color="#2196F3", edgecolors="white", s=60)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs. Predicted")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # --- Chart 2: Residuals (errors) ---
    ax = axes[1]
    residuals = y_test.values - y_pred
    ax.hist(residuals, bins=25, color="#4CAF50", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Prediction Error ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Errors (Residuals)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # --- Chart 3: Price vs Size ---
    ax = axes[2]
    scatter = ax.scatter(df["size_sqft"], df["price"],
                         c=df["bedrooms"], cmap="viridis", alpha=0.7, s=40)
    plt.colorbar(scatter, ax=ax, label="Bedrooms")
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Price ($)")
    ax.set_title("Price vs. Size (colored by bedrooms)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    plt.tight_layout()
    plt.savefig("house_price_results.png", dpi=120, bbox_inches="tight")
    print("\nCharts saved to house_price_results.png")
    plt.show()


# ── 4. Predict new houses ────────────────────────────────────

def predict_houses(model, scaler):
    """Predict prices for a few new houses."""
    new_houses = pd.DataFrame({
        "size_sqft": [1200, 2500, 800,  3500],
        "bedrooms":  [2,    4,    1,    5   ],
        "age_years": [5,    10,   30,   2   ],
        "distance_km": [3.0, 8.5, 20.0, 1.5],
    })

    scaled = scaler.transform(new_houses)
    predictions = model.predict(scaled)

    print("\n" + "=" * 70)
    print("  PREDICTING PRICES FOR NEW HOUSES")
    print("=" * 70)
    header = f"  {'Size':>6}  {'Beds':>4}  {'Age':>4}  {'Dist':>6}  {'Predicted Price':>16}"
    print(header)
    print("-" * 70)
    for (_, row), pred in zip(new_houses.iterrows(), predictions):
        print(f"  {row['size_sqft']:>5.0f}  {row['bedrooms']:>4.0f}  "
              f"{row['age_years']:>4.0f}  {row['distance_km']:>5.1f}km  "
              f"  ${pred:>14,.0f}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic housing dataset...")
    df = generate_dataset(n_samples=300)

    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nBasic statistics:")
    print(df.describe().to_string())

    model, scaler, X_test, y_test, y_pred = train_model(df)
    plot_results(y_test, y_pred, df)
    predict_houses(model, scaler)

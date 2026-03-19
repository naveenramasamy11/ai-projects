"""
Weather Classifier
==================
A beginner-friendly machine learning project that predicts the type of weather
(Sunny, Rainy, Cloudy, Snowy) based on meteorological features such as
temperature, humidity, wind speed, and cloud cover.

Key Concepts:
- Random Forest Classification
- Feature Engineering (creating new features from existing ones)
- Train/test splitting and cross-validation
- Feature importance visualization
- Classification report and confusion matrix

Dataset: Synthetic dataset generated programmatically — no downloads needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_weather_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic weather dataset.

    Each row represents one day's observations. Weather type is derived from
    the raw features using realistic meteorological rules so that the
    classifier has a genuine pattern to learn.

    Parameters
    ----------
    n_samples : int
        Number of daily observations to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and a 'weather_type' target column.
    """
    rng = np.random.default_rng(random_state)

    # Raw features
    temperature = rng.uniform(-10, 40, n_samples)   # Celsius
    humidity    = rng.uniform(10, 100, n_samples)    # Percentage
    wind_speed  = rng.uniform(0, 80, n_samples)      # km/h
    cloud_cover = rng.uniform(0, 100, n_samples)     # Percentage
    pressure    = rng.uniform(970, 1040, n_samples)  # hPa

    # ── Feature Engineering: create derived features ──────────────────────
    # Heat index: feels-hotter-than-temp indicator
    heat_index = temperature + 0.33 * (humidity / 100 * 6.105) - 4

    # Discomfort score: combines heat and wind
    discomfort = heat_index - 0.55 * (1 - humidity / 100) * (heat_index - 14.5)

    # ── Rule-based label assignment (mimics real meteorology) ─────────────
    labels = []
    for temp, hum, wind, cloud, pres in zip(
            temperature, humidity, wind_speed, cloud_cover, pressure):

        if temp <= 2 and hum > 60 and cloud > 70:
            label = "Snowy"
        elif hum > 70 and cloud > 60 and pres < 1005:
            label = "Rainy"
        elif cloud > 50 and hum > 50:
            label = "Cloudy"
        else:
            label = "Sunny"

        labels.append(label)

    df = pd.DataFrame({
        "temperature_c":  temperature,
        "humidity_pct":   humidity,
        "wind_speed_kmh": wind_speed,
        "cloud_cover_pct": cloud_cover,
        "pressure_hpa":   pressure,
        "heat_index":     heat_index,
        "discomfort":     discomfort,
        "weather_type":   labels,
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Encode labels, split data, train a Random Forest, and return results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset returned by generate_weather_data().

    Returns
    -------
    tuple
        (model, X_test, y_test, y_pred, label_encoder, feature_names)
    """
    feature_cols = [c for c in df.columns if c != "weather_type"]

    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["weather_type"])   # e.g. Cloudy→0, Rainy→1 …

    # 80 / 20 train-test split, stratified to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest: an ensemble of decision trees
    model = RandomForestClassifier(
        n_estimators=100,   # number of trees
        max_depth=8,        # limit tree depth to prevent overfitting
        random_state=42,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, le, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(model, X_test, y_test, y_pred, le, feature_cols: list,
                 output_path: str = "weather_classifier_results.png") -> None:
    """
    Create a 2×2 figure with:
      - Confusion matrix heatmap
      - Feature importances bar chart
      - Weather type distribution
      - Per-class precision / recall bar chart

    Parameters
    ----------
    output_path : str
        File path where the PNG is saved.
    """
    class_names = le.classes_

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weather Classifier — Random Forest Results", fontsize=16, fontweight="bold")

    # ── 1. Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # ── 2. Feature importance ─────────────────────────────────────────────
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    feat_series.plot(kind="barh", ax=axes[0, 1], color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Feature Importances")
    axes[0, 1].set_xlabel("Importance Score")

    # ── 3. Weather type distribution in test set ──────────────────────────
    actual_labels = le.inverse_transform(y_test)
    pd.Series(actual_labels).value_counts().plot(
        kind="bar", ax=axes[1, 0], color=["gold", "steelblue", "grey", "skyblue"],
        edgecolor="black", rot=0
    )
    axes[1, 0].set_title("Weather Type Distribution (Test Set)")
    axes[1, 0].set_xlabel("Weather Type")
    axes[1, 0].set_ylabel("Count")

    # ── 4. Per-class precision & recall ───────────────────────────────────
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, labels=range(len(class_names)))
    x = np.arange(len(class_names))
    w = 0.35
    axes[1, 1].bar(x - w/2, prec, w, label="Precision", color="cornflowerblue")
    axes[1, 1].bar(x + w/2, rec,  w, label="Recall",    color="salmon")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_title("Per-Class Precision & Recall")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full weather-classifier pipeline."""

    print("=" * 55)
    print("  Weather Classifier — Random Forest Demo")
    print("=" * 55)

    # Step 1: Generate data
    print("\n[1/4] Generating synthetic weather dataset …")
    df = generate_weather_data(n_samples=1000)
    print(f"      {len(df)} rows | {df['weather_type'].value_counts().to_dict()}")

    # Step 2: Train model
    print("\n[2/4] Training Random Forest classifier …")
    model, X_test, y_test, y_pred, le, feature_cols = train_model(df)

    # Step 3: Evaluate
    print("\n[3/4] Evaluation results:")
    acc = accuracy_score(y_test, y_pred)
    print(f"      Accuracy : {acc:.2%}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation (5-fold) on the full dataset for a robust estimate
    X_all = df[[c for c in df.columns if c != "weather_type"]].values
    y_all = le.transform(df["weather_type"])
    cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring="accuracy")
    print(f"      5-Fold CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Step 4: Visualise
    print("\n[4/4] Generating charts …")
    plot_results(model, X_test, y_test, y_pred, le, feature_cols)

    # Quick prediction demo
    print("\n--- Quick Prediction Demo ---")
    sample = pd.DataFrame([{
        "temperature_c": 28, "humidity_pct": 45, "wind_speed_kmh": 15,
        "cloud_cover_pct": 20, "pressure_hpa": 1018,
        "heat_index": 28 + 0.33 * (45/100 * 6.105) - 4,
        "discomfort": 0,  # simplified
    }])
    # fix discomfort value
    hi = sample["heat_index"].values[0]
    sample["discomfort"] = hi - 0.55 * (1 - 45/100) * (hi - 14.5)

    pred_label = le.inverse_transform(model.predict(sample[feature_cols]))[0]
    proba = model.predict_proba(sample[feature_cols])[0]
    print(f"  Input  : 28°C, 45% humidity, 15 km/h wind, 20% cloud, 1018 hPa")
    print(f"  Prediction : {pred_label}")
    print(f"  Probabilities : { {c: f'{p:.0%}' for c, p in zip(le.classes_, proba)} }")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

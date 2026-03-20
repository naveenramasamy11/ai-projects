"""
customer_segmentation.py
========================
Project 10 — Customer Segmentation using K-Means Clustering

What this project does
-----------------------
We take a synthetic dataset of retail customers described by two features —
Annual Spending Score and Annual Income — and group them into meaningful
segments using the K-Means clustering algorithm.

Key concepts you will see here:
- Unsupervised learning: there are NO labels; the algorithm finds structure itself.
- K-Means: assigns each data point to the nearest cluster centroid and iteratively
  refines the centroids to minimise within-cluster variance.
- The Elbow Method: a technique to choose the right number of clusters (k) by
  plotting inertia (sum of squared distances to nearest centroid) vs. k.
- Feature scaling: K-Means uses Euclidean distance, so features should be on the
  same scale before clustering.

Run this file directly:
    python customer_segmentation.py

It will:
1. Generate a synthetic dataset of 300 customers.
2. Find the optimal k using the Elbow Method.
3. Fit K-Means with k=5.
4. Print a summary of each cluster.
5. Save a cluster plot to  customer_segments.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# 1. Data Generation
# ---------------------------------------------------------------------------

def generate_customer_data(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic customer dataset with five natural clusters.

    Parameters
    ----------
    n_samples : int
        Total number of customer records to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CustomerID, AnnualIncome_k, SpendingScore.
        - AnnualIncome_k  : annual income in thousands of dollars (15 – 135).
        - SpendingScore   : spending score from 1 (low) to 100 (high), assigned
                            by the retailer based on purchase behaviour.
    """
    rng = np.random.default_rng(random_state)

    # Define five cluster centres [income, spending]
    centres = [
        [25, 78],   # Low income, high spenders  (impulsive buyers)
        [55, 55],   # Mid income, mid spenders    (balanced)
        [85, 20],   # High income, low spenders   (careful savers)
        [85, 82],   # High income, high spenders  (premium shoppers)
        [30, 25],   # Low income, low spenders    (budget-conscious)
    ]
    cluster_sizes = [n_samples // 5] * 4 + [n_samples - 4 * (n_samples // 5)]

    records = []
    for idx, (centre, size) in enumerate(zip(centres, cluster_sizes)):
        income = rng.normal(loc=centre[0], scale=8, size=size).clip(15, 135)
        score  = rng.normal(loc=centre[1], scale=8, size=size).clip(1, 100)
        for i_val, s_val in zip(income, score):
            records.append({"AnnualIncome_k": round(i_val, 1),
                             "SpendingScore": round(s_val, 1)})

    df = pd.DataFrame(records)
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature Scaling
# ---------------------------------------------------------------------------

def scale_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Standardise selected columns to zero mean and unit variance.

    K-Means relies on Euclidean distance, so large-scale features (e.g. income
    in thousands) would otherwise dominate small-scale features (score 1-100).

    Parameters
    ----------
    df           : Customer DataFrame.
    feature_cols : List of column names to scale.

    Returns
    -------
    np.ndarray   : Scaled feature matrix.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df[feature_cols])


# ---------------------------------------------------------------------------
# 3. Elbow Method — Choose k
# ---------------------------------------------------------------------------

def elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Compute inertia and silhouette score for a range of k values.

    Inertia   : sum of squared distances from each point to its cluster centre.
                Lower is better, but always decreases as k increases.
    Silhouette: measures how similar a point is to its own cluster vs others.
                Ranges from -1 to +1; higher is better.

    The 'elbow' in the inertia curve is where additional clusters give
    diminishing returns — that k is usually a good choice.

    Parameters
    ----------
    X_scaled : Scaled feature matrix.
    k_range  : Range of k values to evaluate.

    Returns
    -------
    dict with keys 'k_values', 'inertias', 'silhouettes'.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return {"k_values": list(k_range),
            "inertias": inertias,
            "silhouettes": silhouettes}


# ---------------------------------------------------------------------------
# 4. Fit K-Means
# ---------------------------------------------------------------------------

def fit_kmeans(X_scaled: np.ndarray, k: int = 5) -> KMeans:
    """
    Fit K-Means with the chosen number of clusters.

    Parameters
    ----------
    X_scaled : Scaled feature matrix.
    k        : Number of clusters.

    Returns
    -------
    Fitted KMeans object.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


# ---------------------------------------------------------------------------
# 5. Cluster Summary
# ---------------------------------------------------------------------------

def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a human-readable summary of each cluster.

    Parameters
    ----------
    df : DataFrame with an added 'Cluster' column.

    Returns
    -------
    pd.DataFrame summarising mean income, mean score, and size per cluster.
    """
    summary = (df.groupby("Cluster")
                 .agg(Count=("CustomerID", "count"),
                      Avg_Income=("AnnualIncome_k", "mean"),
                      Avg_Score=("SpendingScore", "mean"))
                 .round(1)
                 .reset_index())

    # Assign a descriptive label based on income / spending quartiles
    labels_map = {
        0: "Low Income / High Spenders",
        1: "Mid Income / Mid Spenders",
        2: "High Income / Low Spenders",
        3: "High Income / High Spenders",
        4: "Low Income / Low Spenders",
    }
    # Sort by average income then spending to give consistent label assignment
    summary = summary.sort_values(["Avg_Income", "Avg_Score"]).reset_index(drop=True)
    summary["Segment_Label"] = [
        "Budget Conscious",
        "Low Income Impulse Buyers",
        "Middle of the Road",
        "Careful High Earners",
        "Premium Shoppers",
    ][:len(summary)]

    return summary


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def plot_clusters(df: pd.DataFrame, output_path: str = "customer_segments.png") -> None:
    """
    Draw two side-by-side plots:
      Left  — Elbow curve (inertia vs k).
      Right — Scatter plot of customers coloured by cluster.

    Parameters
    ----------
    df          : DataFrame with 'AnnualIncome_k', 'SpendingScore', 'Cluster'.
    output_path : File path where the PNG will be saved.
    """
    elbow_data = elbow_method(
        scale_features(df, ["AnnualIncome_k", "SpendingScore"])
    )

    palette = sns.color_palette("tab10", n_colors=df["Cluster"].nunique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Customer Segmentation — K-Means Clustering", fontsize=15, fontweight="bold")

    # --- Elbow Curve ---
    ax1 = axes[0]
    ax1.plot(elbow_data["k_values"], elbow_data["inertias"],
             marker="o", color="steelblue", linewidth=2, markersize=7)
    ax1.axvline(x=5, color="tomato", linestyle="--", linewidth=1.5, label="Chosen k=5")
    ax1.set_title("Elbow Method — Choosing k", fontsize=13)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia (Within-cluster Sum of Squares)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Customer Scatter ---
    ax2 = axes[1]
    for cluster_id, group in df.groupby("Cluster"):
        ax2.scatter(group["AnnualIncome_k"], group["SpendingScore"],
                    label=f"Cluster {cluster_id}",
                    color=palette[cluster_id], alpha=0.75, s=60, edgecolors="white", linewidths=0.4)
    ax2.set_title("Customer Segments by Income & Spending Score", fontsize=13)
    ax2.set_xlabel("Annual Income (k$)", fontsize=11)
    ax2.set_ylabel("Spending Score (1–100)", fontsize=11)
    ax2.legend(title="Cluster", fontsize=9, title_fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Project 10 — Customer Segmentation (K-Means)")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic customer dataset …")
    df = generate_customer_data(n_samples=300)
    print(f"      {len(df)} customers generated.")
    print(df.head(5).to_string(index=False))

    # Step 2: Scale features
    print("\n[2/5] Scaling features with StandardScaler …")
    feature_cols = ["AnnualIncome_k", "SpendingScore"]
    X_scaled = scale_features(df, feature_cols)
    print("      Features scaled to zero mean and unit variance.")

    # Step 3: Elbow method
    print("\n[3/5] Running Elbow Method (k = 2 … 10) …")
    elbow = elbow_method(X_scaled)
    print(f"      {'k':>3}  {'Inertia':>10}  {'Silhouette':>12}")
    print("      " + "-" * 30)
    for k, ine, sil in zip(elbow["k_values"], elbow["inertias"], elbow["silhouettes"]):
        marker = " ← chosen" if k == 5 else ""
        print(f"      {k:>3}  {ine:>10.1f}  {sil:>12.4f}{marker}")

    # Step 4: Fit final model
    print("\n[4/5] Fitting K-Means with k=5 …")
    km_model = fit_kmeans(X_scaled, k=5)
    df["Cluster"] = km_model.labels_
    sil = silhouette_score(X_scaled, km_model.labels_)
    print(f"      Silhouette Score (k=5): {sil:.4f}  (higher = better separated clusters)")

    # Step 5: Summary & visualisation
    print("\n[5/5] Cluster Summary:")
    summary = cluster_summary(df)
    print(summary.to_string(index=False))

    print("\n      Saving visualisation …")
    plot_clusters(df, output_path="customer_segments.png")

    print("\n✅  Done! Run `python customer_segmentation.py` any time to reproduce results.")
    print("=" * 60)

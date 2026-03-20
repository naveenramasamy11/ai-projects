# 🛍️ Project 10 — Customer Segmentation

> Group retail customers into meaningful segments using **K-Means Clustering**, an unsupervised machine learning algorithm.

---

## File Table

| File | Purpose |
|---|---|
| `customer_segmentation.py` | Self-contained Python script — generates data, clusters it, prints summary, saves a chart |
| `customer_segmentation.ipynb` | Step-by-step Jupyter notebook with explanations and experiments |
| `requirements.txt` | Pinned pip dependencies |
| `customer_segments.png` | Output chart (created when you run the script or notebook) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python customer_segmentation.py
```

You should see console output followed by a saved `customer_segments.png`.

To open the notebook:
```bash
jupyter notebook customer_segmentation.ipynb
```

---

## Example Output

```
============================================================
  Project 10 — Customer Segmentation (K-Means)
============================================================

[1/5] Generating synthetic customer dataset …
      300 customers generated.

[3/5] Running Elbow Method (k = 2 … 10) …
        k     Inertia    Silhouette
        2    652.3       0.4312
        3    420.1       0.5108
        4    301.7       0.5534
        5    198.4       0.6241 ← chosen
        6    185.2       0.5980

[4/5] Fitting K-Means with k=5 …
      Silhouette Score (k=5): 0.6241  (higher = better separated clusters)

[5/5] Cluster Summary:
 Cluster  Count  Avg_Income  Avg_Score  Segment_Label
       0     61        26.1       78.2  Low Income Impulse Buyers
       1     59        55.4       54.8  Middle of the Road
       2     60        85.3       20.1  Careful High Earners
       3     61        84.9       82.4  Premium Shoppers
       4     59        29.8       25.3  Budget Conscious

  Chart saved → customer_segments.png
✅  Done!
```

---

## Key Concepts

| Concept | Plain English |
|---|---|
| **Unsupervised Learning** | Finding structure in data without predefined labels |
| **K-Means** | Iteratively assigns points to the nearest centroid and recomputes centroids |
| **Inertia** | Sum of squared distances from each point to its cluster centre |
| **Elbow Method** | Plot inertia vs k and look for the bend — that k is usually best |
| **Silhouette Score** | How well-separated clusters are; ranges –1 to +1 (higher = better) |
| **Feature Scaling** | Puts all features on the same scale so distance isn't biased |
| **StandardScaler** | Transforms features to mean=0, std=1 |

---

## Libraries Used

| Library | Purpose | Link |
|---|---|---|
| `scikit-learn` | KMeans, StandardScaler, silhouette_score | [scikit-learn.org](https://scikit-learn.org) |
| `numpy` | Random data generation, array maths | [numpy.org](https://numpy.org) |
| `pandas` | DataFrame manipulation and group summaries | [pandas.pydata.org](https://pandas.pydata.org) |
| `matplotlib` | Elbow curve and scatter plot | [matplotlib.org](https://matplotlib.org) |
| `seaborn` | Plot styling and colour palettes | [seaborn.pydata.org](https://seaborn.pydata.org) |

---

*Part of the [ai-projects](https://github.com/naveenramasamy11/ai-projects) beginner AI/ML learning series.*

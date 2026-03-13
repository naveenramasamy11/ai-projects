"""
Iris Flower Classifier
======================
A beginner-friendly introduction to multi-class classification using the
famous Iris dataset. We compare two classic algorithms:
  - K-Nearest Neighbors (KNN)
  - Decision Tree

The Iris dataset contains 150 samples of iris flowers across 3 species
(Setosa, Versicolor, Virginica) described by 4 features:
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
  - petal width (cm)

Learning Goals:
  - Understand train/test splits
  - Fit and evaluate a KNN classifier
  - Fit and evaluate a Decision Tree classifier
  - Visualise a confusion matrix
  - Plot a decision boundary

Run:
    python iris_flower_classifier.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load the Iris dataset and return X, y, and feature/target names."""
    iris = load_iris()
    X = iris.data          # shape (150, 4)
    y = iris.target        # 0 = Setosa, 1 = Versicolor, 2 = Virginica
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def split_and_scale(X, y, test_size=0.25, random_state=42):
    """
    Split data into train/test sets and standardise features.

    Standardisation (mean=0, std=1) helps KNN, which is distance-based,
    treat all features equally regardless of their original scale.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier.

    KNN classifies a new sample by looking at the k closest training samples
    and taking a majority vote.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours to consider (k).
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def train_decision_tree(X_train, y_train, max_depth=4, random_state=42):
    """
    Train a Decision Tree classifier.

    A Decision Tree learns a hierarchy of if/else rules from the training
    data. `max_depth` limits tree complexity to avoid overfitting.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


def evaluate_model(model, X_test, y_test, target_names, model_name):
    """
    Print accuracy and a detailed classification report for a fitted model.

    Returns predicted labels for further use.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.2%}")
    print()
    print(classification_report(y_test, y_pred, target_names=target_names))
    return y_pred


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, target_names, title, ax):
    """
    Draw a heatmap confusion matrix on the given Axes object.

    The matrix shows how many samples of each true class were predicted
    as each class. Perfect predictions land on the diagonal.
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")


def plot_decision_boundary(model, X_scaled, y, feature_names, target_names, ax):
    """
    Visualise the decision boundary for the first two (scaled) features.

    We create a mesh of points covering the feature space, predict the
    class for each point, and colour-code the regions.

    Note: This uses only 2 features (sepal length & sepal width) so the
    boundary is an approximation – the model is trained on all 4.
    """
    X2 = X_scaled[:, :2]   # petal length & petal width (features 2 & 3 are
                             # most discriminative, but we keep sepal for variety)

    # Re-fit on 2 features just for visualisation
    from sklearn.base import clone
    model_2d = clone(model)
    model_2d.fit(X2, y)

    h = 0.02  # mesh step size
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = plt.cm.RdYlBu
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    colors = ["#e6194b", "#3cb44b", "#4363d8"]
    for cls_idx, cls_name in enumerate(target_names):
        mask = y == cls_idx
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=colors[cls_idx], label=cls_name,
            edgecolors="k", s=40, alpha=0.8,
        )
    ax.set_xlabel(f"{feature_names[0]} (scaled)", fontsize=10)
    ax.set_ylabel(f"{feature_names[1]} (scaled)", fontsize=10)
    ax.legend(fontsize=9)


def save_all_figures(
    knn, dt,
    X_train_scaled, X_test_scaled,
    y_train, y_test,
    y_pred_knn, y_pred_dt,
    feature_names, target_names,
    output_path="iris_results.png",
):
    """
    Compose a 2×2 figure and save it to disk.

    Panels:
      [0,0] KNN confusion matrix
      [0,1] Decision Tree confusion matrix
      [1,0] KNN decision boundary (2 features)
      [1,1] Decision Tree visualisation (tree diagram)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Iris Flower Classifier — KNN vs Decision Tree", fontsize=16, y=1.01)

    # --- Confusion matrices ---
    plot_confusion_matrix(y_test, y_pred_knn, target_names, "KNN — Confusion Matrix",       axes[0, 0])
    plot_confusion_matrix(y_test, y_pred_dt,  target_names, "Decision Tree — Confusion Matrix", axes[0, 1])

    # --- Decision boundaries (using full training set for context) ---
    X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
    y_all        = np.concatenate([y_train, y_test])

    plot_decision_boundary(knn, X_all_scaled, y_all, feature_names, target_names, axes[1, 0])
    axes[1, 0].set_title("KNN — Decision Boundary (2 features)", fontsize=12, fontweight="bold")

    # --- Decision tree diagram ---
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Decision Tree Structure (max_depth=4)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\n📊 Figure saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("🌸 Iris Flower Classifier")
    print("   Comparing KNN vs Decision Tree\n")

    # 1. Load data
    X, y, feature_names, target_names = load_data()
    print(f"Dataset shape : {X.shape}")
    print(f"Classes       : {list(target_names)}")
    print(f"Features      : {list(feature_names)}")

    # 2. Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"\nTrain samples : {len(y_train)}")
    print(f"Test samples  : {len(y_test)}")

    # 3. Train models
    knn = train_knn(X_train, y_train, n_neighbors=5)
    dt  = train_decision_tree(X_train, y_train, max_depth=4)

    # 4. Evaluate
    y_pred_knn = evaluate_model(knn, X_test, y_test, target_names, "K-Nearest Neighbors (k=5)")
    y_pred_dt  = evaluate_model(dt,  X_test, y_test, target_names, "Decision Tree (max_depth=4)")

    # 5. Predict a new sample
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # looks like Setosa
    new_scaled = scaler.transform(new_sample)
    knn_pred = target_names[knn.predict(new_scaled)[0]]
    dt_pred  = target_names[dt.predict(new_scaled)[0]]
    print(f"\n🌼 New sample {new_sample[0].tolist()}")
    print(f"   KNN predicts      : {knn_pred}")
    print(f"   Decision Tree predicts : {dt_pred}")

    # 6. Save figures
    save_all_figures(
        knn, dt,
        X_train, X_test,
        y_train, y_test,
        y_pred_knn, y_pred_dt,
        feature_names, target_names,
        output_path="iris_results.png",
    )

    print("\n✅ Done! Check iris_results.png for visualisations.")


if __name__ == "__main__":
    main()

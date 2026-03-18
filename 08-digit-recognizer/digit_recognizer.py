"""
Digit Recognizer — Project 08
==============================
Train a neural network (Multi-Layer Perceptron) to recognize handwritten digits
using the built-in MNIST-like dataset from scikit-learn (load_digits, 8×8 images).

Key concepts covered:
- Image data as pixel arrays
- Train/test split
- Feature scaling (StandardScaler)
- Multi-Layer Perceptron (MLP) — a basic neural network
- Multi-class classification
- Confusion matrix & classification report
- Saving a trained model with joblib
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# ---------------------------------------------------------------------------
# 1. Load and explore the dataset
# ---------------------------------------------------------------------------

def load_data():
    """
    Load the built-in digits dataset from scikit-learn.

    Returns
    -------
    X : ndarray of shape (1797, 64)
        Flattened 8×8 pixel images (values 0–16).
    y : ndarray of shape (1797,)
        Target digit labels (0–9).
    images : ndarray of shape (1797, 8, 8)
        Original 2-D image arrays for visualisation.
    """
    digits = load_digits()
    print(f"Dataset loaded: {digits.data.shape[0]} samples, "
          f"{digits.data.shape[1]} features (8×8 pixels), "
          f"{len(digits.target_names)} classes (0–9)")
    return digits.data, digits.target, digits.images


# ---------------------------------------------------------------------------
# 2. Visualise sample digits
# ---------------------------------------------------------------------------

def plot_sample_digits(images, labels, n=20, save_path="sample_digits.png"):
    """
    Display the first *n* digit images in a grid and save to a PNG file.

    Parameters
    ----------
    images : ndarray of shape (N, 8, 8)
    labels : ndarray of shape (N,)
    n      : int — number of samples to show (must be <= len(images))
    save_path : str — output PNG filename
    """
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i], cmap="gray_r", interpolation="nearest")
        axes[i].set_title(f"Label: {labels[i]}", fontsize=8)
        axes[i].axis("off")

    # Hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Handwritten Digits from the Dataset", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Sample digits plot saved → {save_path}")


# ---------------------------------------------------------------------------
# 3. Preprocess: scale features
# ---------------------------------------------------------------------------

def preprocess(X_train, X_test):
    """
    Standardise pixel values so each feature has mean=0, std=1.
    Fit the scaler on the training set only to avoid data leakage.

    Parameters
    ----------
    X_train, X_test : ndarray

    Returns
    -------
    X_train_scaled, X_test_scaled : ndarray
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 4. Build and train the MLP (neural network)
# ---------------------------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train a Multi-Layer Perceptron classifier.

    Architecture: two hidden layers of 128 and 64 neurons.

    Parameters
    ----------
    X_train : ndarray — scaled training features
    y_train : ndarray — training labels

    Returns
    -------
    model : trained MLPClassifier
    """
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # two hidden layers
        activation="relu",             # ReLU activation function
        solver="adam",                 # adaptive learning-rate optimiser
        max_iter=500,                  # maximum training epochs
        random_state=42,
        verbose=False
    )
    print("Training the neural network (MLP)…")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# ---------------------------------------------------------------------------
# 5. Evaluate the model
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Print accuracy, full classification report, and return predictions.

    Parameters
    ----------
    model   : trained classifier
    X_test  : ndarray — scaled test features
    y_test  : ndarray — true labels

    Returns
    -------
    y_pred : ndarray — predicted labels
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    return y_pred


# ---------------------------------------------------------------------------
# 6. Plot confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, save_path="digit_confusion_matrix.png"):
    """
    Plot and save the confusion matrix as a heatmap.

    Parameters
    ----------
    y_test    : ndarray — true labels
    y_pred    : ndarray — predicted labels
    save_path : str — output PNG filename
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.title("Confusion Matrix — Digit Recognizer", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# 7. Show a few predictions vs actual
# ---------------------------------------------------------------------------

def plot_predictions(images_test, y_test, y_pred, n=12, save_path="digit_predictions.png"):
    """
    Visualise *n* test images with their true and predicted labels.
    Correct predictions are shown in green, wrong ones in red.

    Parameters
    ----------
    images_test : ndarray of shape (M, 8, 8) — test set images
    y_test      : ndarray — true labels
    y_pred      : ndarray — predicted labels
    n           : int — number of samples to display
    save_path   : str — output PNG filename
    """
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2))
    axes = axes.flatten()

    for i in range(n):
        color = "green" if y_pred[i] == y_test[i] else "red"
        axes[i].imshow(images_test[i], cmap="gray_r", interpolation="nearest")
        axes[i].set_title(
            f"True: {y_test[i]}\nPred: {y_pred[i]}",
            color=color,
            fontsize=9
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Prediction samples saved → {save_path}")


# ---------------------------------------------------------------------------
# Main — run the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Load data
    X, y, images = load_data()

    # Step 2: Visualise a few samples
    plot_sample_digits(images, y, n=20, save_path="sample_digits.png")

    # Step 3: Split into training (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Also keep the corresponding 2-D images for visualisation later
    _, images_test, _, _ = train_test_split(
        images, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    # Step 5: Train neural network
    model = train_model(X_train_scaled, y_train)

    # Step 6: Evaluate
    y_pred = evaluate_model(model, X_test_scaled, y_test)

    # Step 7: Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path="digit_confusion_matrix.png")

    # Step 8: Show sample predictions
    plot_predictions(images_test, y_test, y_pred, n=12, save_path="digit_predictions.png")

    # Step 9: Save the trained model for future use
    joblib.dump(model, "digit_recognizer_model.pkl")
    joblib.dump(scaler, "digit_scaler.pkl")
    print("\nModel saved → digit_recognizer_model.pkl")
    print("Scaler saved → digit_scaler.pkl")

    print("\n✅ All done! Check the PNG files for visualisations.")

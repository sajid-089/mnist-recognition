# ============================================================
# MNIST HANDWRITTEN DIGIT RECOGNITION
# Machine Learning Internship Project
# ============================================================

# --- importing required libraries ---
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

def load_data():
    """
    Load the MNIST dataset using sklearn's fetch_openml function.
    MNIST contains 70,000 grayscale images of handwritten digits (0-9).
    Each image is 28x28 pixels, flattened into 784 features.
    This function downloads the dataset on first run and caches it locally.
    """
    print("\n[INFO] Loading MNIST dataset...")
    print("  (first time download may take 1-2 minutes)")

    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)

    images = mnist_data.data
    labels = mnist_data.target.astype(int)

    print(f"  Total images: {images.shape[0]}")
    print(f"  Features per image: {images.shape[1]} (28x28 = 784 pixels))
    print(f"  Digit classes: {np.unique(labels)}")
    print(f"  Pixel value range: {images.min()} to {images.max()}")

    return images, labels


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

def explore_data(images, labels):
    """
    Analyze the dataset to understand class distribution.
    A balanced dataset ensures the model gets equal exposure to all digits.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    print("\nDigit distribution across entire dataset:")
    unique, counts = np.unique(labels, return_counts=True)
    for digit, count in zip(unique, counts):
        bar = '|' * (count // 300)
        print(f"  Digit {digit}: {count:>5} {bar}")

    return unique, counts


def plot_sample_images(images, labels):
    """
    Display 15 random handwritten digit images from the dataset.
    Each image needs to be reshaped from 784-length vector back to 28x28 grid.
    """
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(15):
        idx = np.random.randint(0, len(images))
        img = images[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {labels[idx]}", fontsize=11)
        axes[i].axis('off')

    plt.suptitle("Sample Images from MNIST Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Sample images saved as 'sample_digits.png'")


def plot_digit_distribution(labels):
    """
    Create a bar chart showing the number of samples for each digit.
    Helps verify that no digit class is significantly underrepresented.
    """
    plt.figure(figsize=(10, 5))
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    plt.bar(unique, counts, color=colors)
    plt.title('Digit Distribution in Dataset', fontsize=14)
    plt.xlabel('Digit', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(10))

    for d, c in zip(unique, counts):
        plt.text(d, c + 100, str(c), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('digit_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Distribution plot saved as 'digit_distribution.png'")


def plot_single_digit_detail(images, labels):
    """
    Show a single digit image alongside its pixel intensity heatmap.
    This illustrates how an image is represented as a grid of numbers
    where 0 means black (background) and 255 means white (ink).
    """
    idx = np.where(labels == 5)[0][0]
    img = images[idx].reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Digit: {labels[idx]} (Grayscale)", fontsize=13)
    axes[0].axis('off')

    im = axes[1].imshow(img, cmap='hot')
    axes[1].set_title("Pixel Intensity Heatmap", fontsize=13)
    plt.colorbar(im, ax=axes[1], label='Intensity (0-255)')

    plt.tight_layout()
    plt.savefig('digit_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Digit detail saved as 'digit_detail.png'")


# ============================================================
# STEP 3: PREPROCESSING
# ============================================================

def preprocess_data(images, labels):
    """
    Prepare the data for machine learning models:

    1. Subsample the dataset:
       - Full 70k images would make SVM training extremely slow
       - Using 15,000 images gives good results with reasonable training time
       - This is a common practice when working with large datasets locally

    2. Normalize pixel values:
       - Raw pixels range from 0 to 255
       - Dividing by 255 scales them to 0-1 range
       - Models converge faster with normalized inputs

    3. Apply StandardScaler:
       - Centers data around zero with unit variance
       - Particularly important for SVM which is sensitive to feature scales

    4. Train-test split:
       - 80% for training, 20% for testing
       - stratify ensures each digit has proportional representation in both sets
    """
    print("\n[INFO] Preprocessing data...")

    # subsample for manageable training time
    sample_size = 15000
    rng = np.random.RandomState(42)
    selected = rng.choice(len(images), sample_size, replace=False)
    images_sub = images[selected]
    labels_sub = labels[selected]
    print(f"  Using {sample_size} images for training efficiency")

    # normalize pixel values to 0-1 range
    images_sub = images_sub / 255.0
    print(f"  Normalized pixel range: {images_sub.min():.1f} to {images_sub.max():.1f}")

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        images_sub, labels_sub, test_size=0.2,
        random_state=42, stratify=labels_sub
    )
    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Test set:     {X_test.shape[0]} images")

    # apply standard scaling for better model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"  Applied StandardScaler for feature normalization")

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4: MODEL TRAINING
# ============================================================

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.

    How Random Forest works:
    - Creates multiple decision trees (150 in our case)
    - Each tree is trained on a random subset of the data
    - For prediction, all trees vote and the majority wins
    - This ensemble approach reduces overfitting and improves accuracy

    Parameters:
    - n_estimators=150: number of trees in the forest
    - n_jobs=-1: use all available CPU cores for parallel training
    """
    print("\n[INFO] Training Random Forest classifier...")
    print("  (this may take 1-2 minutes)")

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("  Random Forest training complete!")
    return rf_model


def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine (SVM) classifier.

    How SVM works:
    - Finds the optimal boundary (hyperplane) between different classes
    - The RBF kernel allows it to handle non-linear patterns
    - Very effective for image classification tasks

    Parameters:
    - kernel='rbf': Radial Basis Function kernel for non-linear classification
    - C=5: controls the trade-off between smooth boundary and correct classification
    - gamma='scale': automatically calculates the kernel coefficient
    """
    print("\n[INFO] Training SVM classifier...")
    print("  (this may take 2-4 minutes, please wait)")

    svm_model = SVC(
        kernel='rbf',
        C=5,
        gamma='scale',
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    print("  SVM training complete!")
    return svm_model


# ============================================================
# STEP 5: EVALUATION
# ============================================================

def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    The test set contains images the model has never seen before,
    giving an honest measure of real-world performance.
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Results")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Correctly classified: {int(acc * len(y_test))} out of {len(y_test)}")

    print(f"\n  Per-Digit Classification Report:")
    target_names = [f'Digit {i}' for i in range(10)]
    print(classification_report(y_test, predictions, target_names=target_names))

    return predictions, acc


def plot_confusion_matrices(y_test, rf_pred, svm_pred, rf_acc, svm_acc):
    """
    Plot confusion matrices for both models side by side.
    Diagonal cells show correct predictions.
    Off-diagonal cells reveal which digits get confused with each other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Random Forest matrix
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=range(10), yticklabels=range(10))
    axes[0].set_title(f'Random Forest (Acc: {rf_acc:.2%})', fontsize=13)
    axes[0].set_xlabel('Predicted Digit')
    axes[0].set_ylabel('Actual Digit')

    # SVM matrix
    cm_svm = confusion_matrix(y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=range(10), yticklabels=range(10))
    axes[1].set_title(f'SVM (Acc: {svm_acc:.2%})', fontsize=13)
    axes[1].set_xlabel('Predicted Digit')
    axes[1].set_ylabel('Actual Digit')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrices saved as 'confusion_matrices.png'")

    # identify most confused digit pairs from the better model
    best_cm = cm_svm if svm_acc > rf_acc else cm_rf
    temp = best_cm.copy()
    np.fill_diagonal(temp, 0)
    print("\n  Most confused digit pairs (best model):")
    for _ in range(3):
        pos = np.unravel_index(temp.argmax(), temp.shape)
        if temp[pos] > 0:
            print(f"    Digit {pos[0]} confused with Digit {pos[1]}: {temp[pos]} times")
            temp[pos] = 0


def plot_model_comparison(rf_acc, svm_acc):
    """
    Create a bar chart comparing the accuracy of both models.
    Makes it easy to see which algorithm performed better on this task.
    """
    models = ['Random Forest', 'SVM']
    accuracies = [rf_acc, svm_acc]
    colors = ['#27ae60', '#2980b9']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=colors, width=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{acc:.2%}', ha='center', fontsize=14, fontweight='bold')

    ax.set_ylim(0.9, 1.0)
    ax.set_title('Model Comparison - Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Comparison plot saved as 'model_comparison.png'")


# ============================================================
# STEP 6: VISUALIZE PREDICTIONS
# ============================================================

def plot_predictions(X_test, y_test, y_pred):
    """
    Display 15 random test images with their true and predicted labels.
    Green title indicates a correct prediction.
    Red title indicates an incorrect prediction.
    """
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    y_test_arr = np.array(y_test)
    indices = np.random.choice(len(X_test), 15, replace=False)

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        true_val = y_test_arr[idx]
        pred_val = y_pred[idx]

        axes[i].imshow(img, cmap='gray')

        if true_val == pred_val:
            axes[i].set_title(f"True:{true_val} Pred:{pred_val}",
                             color='green', fontsize=11, fontweight='bold')
        else:
            axes[i].set_title(f"True:{true_val} Pred:{pred_val}",
                             color='red', fontsize=11, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle("Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Sample predictions saved as 'sample_predictions.png'")


def plot_wrong_predictions(X_test, y_test, y_pred):
    """
    Show images that the model classified incorrectly.
    These are often ambiguously written digits that would be
    challenging even for humans to identify correctly.
    """
    y_test_arr = np.array(y_test)
    wrong_idx = np.where(y_pred != y_test_arr)[0]
    print(f"\n  Total misclassified: {len(wrong_idx)} out of {len(y_test_arr)}")

    if len(wrong_idx) == 0:
        print("  Perfect classification - zero errors!")
        return

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    num_show = min(10, len(wrong_idx))
    for i in range(num_show):
        idx = wrong_idx[i]
        img = X_test[idx].reshape(28, 28)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True:{y_test_arr[idx]} Pred:{y_pred[idx]}",
                         color='red', fontsize=11)
        axes[i].axis('off')

    for i in range(num_show, 10):
        axes[i].axis('off')

    plt.suptitle("Misclassified Digits - Where the Model Struggled",
                 fontsize=14, color='red')
    plt.tight_layout()
    plt.savefig('wrong_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Wrong predictions saved as 'wrong_predictions.png'")


# ============================================================
# MAIN FUNCTION - ENTIRE PIPELINE RUNS FROM HERE
# ============================================================

def main():
    print("=" * 60)
    print("  MNIST HANDWRITTEN DIGIT RECOGNITION")
    print("  ML Internship Project")
    print("=" * 60)

    # step 1: load dataset
    images, labels = load_data()

    # step 2: explore the data
    explore_data(images, labels)
    plot_sample_images(images, labels)
    plot_digit_distribution(labels)
    plot_single_digit_detail(images, labels)

    # step 3: preprocess
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)

    # step 4: train both models
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # step 5: evaluate both models
    rf_pred, rf_acc = evaluate_model(rf_model, "Random Forest", X_test, y_test)
    svm_pred, svm_acc = evaluate_model(svm_model, "SVM", X_test, y_test)

    # create evaluation plots
    plot_confusion_matrices(y_test, rf_pred, svm_pred, rf_acc, svm_acc)
    plot_model_comparison(rf_acc, svm_acc)

    # step 6: visualize predictions from the best model
    if svm_acc >= rf_acc:
        best_pred = svm_pred
        best_name = "SVM"
    else:
        best_pred = rf_pred
        best_name = "Random Forest"

    plot_predictions(X_test, y_test, best_pred)
    plot_wrong_predictions(X_test, y_test, best_pred)

    # print final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Random Forest Accuracy: {rf_acc:.2%}")
    print(f"  SVM Accuracy:           {svm_acc:.2%}")
    print(f"  Best Model: {best_name}")
    print(f"\n  Generated Files:")
    print(f"    - sample_digits.png")
    print(f"    - digit_distribution.png")
    print(f"    - digit_detail.png")
    print(f"    - confusion_matrices.png")
    print(f"    - model_comparison.png")
    print(f"    - sample_predictions.png")
    print(f"    - wrong_predictions.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
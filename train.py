"""
CLI training pipeline for FaceAuth.

Usage:
    python train.py --data_dir data --method all
    python train.py --data_dir data --method hog --optimize
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import MODEL_DIR
from features.haar_cascade import extract_haar_features
from features.lbp import extract_lbp_features
from features.hog_features import extract_hog_features
from classification.svm_classifier import train_svm, predict
from evaluation.evaluate import compute_metrics, plot_confusion_matrix, plot_method_comparison

from collections import Counter
import numpy as np


def filter_min_samples(X, y, min_samples=2):
    counts = Counter(y)

    keep_indices = [
        i for i, label in enumerate(y)
        if counts[label] >= min_samples
    ]

    X_filtered = X[keep_indices]
    y_filtered = np.array(y)[keep_indices]

    removed = len(y) - len(y_filtered)

    print(f"Filtered identities with < {min_samples} samples.")
    print(f"Removed samples: {removed}")
    print(f"Remaining identities: {len(set(y_filtered))}")
    print(f"Remaining images: {len(y_filtered)}")

    return X_filtered, y_filtered

SUPPORTED_METHODS = ["haar", "lbp", "hog"]


def collect_dataset(data_dir: str):
    """
    Expected structure:

    data/
      Alice/
        img1.jpg
        img2.jpg
      Bob/
        img1.jpg
        img2.jpg
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths = []
    labels = []

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue

        label = person_dir.name

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() in allowed_ext:
                image_paths.append(str(img_path))
                labels.append(label)

    if not image_paths:
        raise ValueError(f"No images found in dataset directory: {data_dir}")

    return image_paths, labels


def get_extractor(method: str):
    if method == "haar":
        return extract_haar_features
    if method == "lbp":
        return extract_lbp_features
    if method == "hog":
        return extract_hog_features

    raise ValueError(f"Unsupported method: {method}")


def build_feature_matrix(image_paths, labels, method: str):
    extractor = get_extractor(method)

    X = []
    y = []
    skipped = []

    for img_path, label in zip(image_paths, labels):
        try:
            features = extractor(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            skipped.append((img_path, str(e)))

    if len(X) == 0:
        raise RuntimeError(f"No valid features extracted for method '{method}'.")

    X = np.vstack(X).astype(np.float32)
    y = np.array(y)

    print(f"\n[{method}] Extracted features: {X.shape}")
    print(f"[{method}] Valid images: {len(y)}")
    print(f"[{method}] Skipped images: {len(skipped)}")

    if skipped:
        print(f"\n[{method}] First skipped samples:")
        for path, reason in skipped[:10]:
            print(f"  - {path}: {reason}")

    return X, y


def train_and_evaluate_method(image_paths, labels, method: str, optimize: bool):
    print(f"\n==============================")
    print(f"Training method: {method}")
    print(f"==============================")

    X, y = build_feature_matrix(image_paths, labels, method)
    X, y = filter_min_samples(X, y, min_samples=2)

    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        raise ValueError(
            f"Need at least 2 identities to train SVM. Found: {unique_classes}"
        )

    class_counts = {label: int(np.sum(y == label)) for label in unique_classes}
    min_count = min(class_counts.values())

    if min_count < 2:
        raise ValueError(
            "Each identity should have at least 2 valid images for train/test split. "
            f"Class counts: {class_counts}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = train_svm(
        X_train,
        y_train,
        method=method,
        optimize=optimize,
    )

    y_pred, probs = predict(model, X_test)


    labels_sorted = model.label_encoder_.classes_
    metrics = compute_metrics(
    y_test,
    y_pred,
    method=method,
    labels=labels_sorted,
)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=labels_sorted,
        method=method,
        save=True,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SVM face recognition models.")

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Dataset directory. Expected format: data/person_name/images.jpg",
    )

    parser.add_argument(
        "--method",
        default="all",
        choices=["all"] + SUPPORTED_METHODS,
        help="Feature method to train.",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use GridSearchCV for SVM hyperparameter optimization.",
    )

    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    image_paths, labels = collect_dataset(args.data_dir)

    print(f"Total images found: {len(image_paths)}")
    print(f"Total identities: {len(set(labels))}")

    methods = SUPPORTED_METHODS if args.method == "all" else [args.method]

    all_results = {}

    for method in methods:
        try:
            metrics = train_and_evaluate_method(
                image_paths=image_paths,
                labels=labels,
                method=method,
                optimize=args.optimize,
            )
            all_results[method] = metrics
        except Exception as e:
            print(f"\n[ERROR] Failed to train method '{method}': {e}")

    if len(all_results) > 1:
        plot_method_comparison(all_results, save=True)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
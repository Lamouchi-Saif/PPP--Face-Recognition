"""
SVM classifier for face recognition.

Supports training on any feature-vector representation (Haar pixels, LBP,
HOG, or DeepFace embeddings) and persists trained models to disk using joblib.
"""
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SVM_KERNEL, SVM_C, SVM_GAMMA, MODEL_DIR


def _model_path(method: str) -> str:
    return os.path.join(MODEL_DIR, f"svm_{method}.pkl")


def train_svm(
    X: np.ndarray,
    y,
    method: str,
    optimize: bool = False,
) -> Pipeline:
    """
    Train an SVM classifier.

    Parameters
    ----------
    X        : feature matrix, shape (N, D).
    y        : labels (list or array of identity names).
    method   : identifier string used when saving the model file.
    optimize : if True, perform a small grid-search over C and gamma.

    Returns
    -------
    sklearn Pipeline (scaler + SVM) fitted on (X, y).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                        probability=True, class_weight='balanced', max_iter=1000)),
        ]
    )

    if optimize:
        param_grid = {
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", "auto"],
        }
        gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        gs.fit(X, y_enc)
        pipeline = gs.best_estimator_
        print(f"Best params: {gs.best_params_}")
    else:
        pipeline.fit(X, y_enc)

    # Attach the label encoder so predictions are human-readable
    pipeline.label_encoder_ = le

    # Persist
    joblib.dump(pipeline, _model_path(method))
    print(f"Model saved → {_model_path(method)}")
    return pipeline


def load_svm(method: str) -> Pipeline:
    """Load a previously saved SVM pipeline."""
    path = _model_path(method)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at {path}. Train first.")
    return joblib.load(path)


def predict(pipeline: Pipeline, X: np.ndarray):
    """
    Predict identity labels for feature matrix *X*.

    Returns
    -------
    labels : list[str]  – predicted identity names.
    probs  : np.ndarray – confidence scores, shape (N, n_classes).
    """
    y_enc = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    labels = pipeline.label_encoder_.inverse_transform(y_enc)
    return labels.tolist(), probs

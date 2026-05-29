"""
Authentication logic – decide whether an incoming face matches a registered
identity.

Two modes are supported:

1. **SVM mode** (traditional + OpenFace features):
   Pass the feature vector through a trained SVM and check the top-class
   probability against a configurable confidence threshold.

2. **DeepFace / embedding mode**:
   Compare the query embedding against all stored embeddings using cosine
   similarity or Euclidean distance.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COSINE_THRESHOLD, EUCLIDEAN_THRESHOLD
from features.deepface import cosine_similarity, euclidean_distance


AUTH_GRANTED = "ACCESS GRANTED"
AUTH_DENIED = "ACCESS DENIED"


# ── SVM-based authentication ────────────────────────────────────────────────────

def authenticate_svm(pipeline, feature_vector: np.ndarray,
                     confidence_threshold: float = 0.6):
    """
    Authenticate using a trained SVM pipeline.

    Parameters
    ----------
    pipeline           : fitted sklearn Pipeline returned by train_svm().
    feature_vector     : 1-D numpy array (single sample).
    confidence_threshold : minimum predicted probability for the top class.

    Returns
    -------
    dict with keys: decision (str), identity (str | None), confidence (float).
    """
    X = feature_vector.reshape(1, -1)
    probs = pipeline.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    identity = pipeline.label_encoder_.classes_[idx]

    if confidence >= confidence_threshold:
        return {"decision": AUTH_GRANTED, "identity": identity,
                "confidence": confidence}
    return {"decision": AUTH_DENIED, "identity": None,
            "confidence": confidence}


# ── Embedding-based authentication ─────────────────────────────────────────────

def authenticate_embedding(
    query_embedding: np.ndarray,
    db_names,
    db_embeddings: np.ndarray,
    metric: str = "cosine",
    cosine_threshold: float = COSINE_THRESHOLD,
    euclidean_threshold: float = EUCLIDEAN_THRESHOLD,
):
    """
    Authenticate a query embedding against a stored embedding database.

    Parameters
    ----------
    query_embedding     : 1-D float32 array (the probe).
    db_names            : list[str] – identity name for each row in db_embeddings.
    db_embeddings       : 2-D array (N, D) of reference embeddings.
    metric              : 'cosine' or 'euclidean'.
    cosine_threshold    : minimum cosine similarity to grant access.
    euclidean_threshold : maximum Euclidean distance to grant access.

    Returns
    -------
    dict with keys:
        decision (str), identity (str | None), best_score (float),
        best_match_index (int).
    """
    if len(db_names) == 0 or db_embeddings.size == 0:
        return {"decision": AUTH_DENIED, "identity": None,
                "best_score": None, "best_match_index": -1}

    if metric == "cosine":
        scores = np.array(
            [cosine_similarity(query_embedding, db_embeddings[i])
             for i in range(len(db_names))]
        )
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        granted = best_score >= cosine_threshold
    else:  # euclidean
        scores = np.array(
            [euclidean_distance(query_embedding, db_embeddings[i])
             for i in range(len(db_names))]
        )
        best_idx = int(np.argmin(scores))
        best_score = float(scores[best_idx])
        granted = best_score <= euclidean_threshold

    identity = db_names[best_idx] if granted else None
    return {
        "decision": AUTH_GRANTED if granted else AUTH_DENIED,
        "identity": identity,
        "best_score": best_score,
        "best_match_index": best_idx,
    }

"""
Local Binary Pattern (LBP) feature extraction.

LBP encodes each pixel as a binary string formed by comparing it with its
circular neighbourhood, then summarises the face image as a concatenated
histogram of these codes – an efficient texture descriptor.
"""
import numpy as np
from skimage.feature import local_binary_pattern
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LBP_RADIUS, LBP_N_POINTS, IMAGE_SIZE
from preprocessing.preprocess import preprocess_image


def extract_lbp_features(
    source,
    radius: int = LBP_RADIUS,
    n_points: int = LBP_N_POINTS,
    n_bins: int = 256,
    target_size=IMAGE_SIZE,
) -> np.ndarray:
    """
    Compute the uniform LBP histogram for a face image.

    Parameters
    ----------
    source      : file path or BGR numpy array.
    radius      : radius of the circular LBP neighbourhood.
    n_points    : number of sampling points on the circle.
    n_bins      : histogram bins (typically n_points + 2 for 'uniform' method).
    target_size : face crop size used during preprocessing.

    Returns
    -------
    np.ndarray, shape (n_bins,) – normalised LBP histogram (float32).
    """
    face = preprocess_image(source, target_size=target_size, grayscale=True)
    if face is None:
        raise ValueError("No face detected in image.")

    # Convert float [0,1] → uint8 for skimage's LBP
    face_uint8 = (face * 255).clip(0, 255).astype(np.uint8)

    lbp = local_binary_pattern(face_uint8, n_points, radius, method="uniform")

    # Build normalised histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7  # L1 normalise

    return hist

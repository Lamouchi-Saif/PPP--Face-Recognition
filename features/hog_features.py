"""
Histogram of Oriented Gradients (HOG) feature extraction.

HOG captures edge structure and shape information from the face image by
computing gradient orientations over small cells and normalising over larger
blocks for illumination invariance.
"""
import numpy as np
from skimage.feature import hog
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    IMAGE_SIZE,
)
from preprocessing.preprocess import preprocess_image


def extract_hog_features(
    source,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
    target_size=IMAGE_SIZE,
) -> np.ndarray:
    """
    Compute the HOG descriptor for a face image.

    Parameters
    ----------
    source          : file path or BGR numpy array.
    orientations    : number of gradient orientation bins.
    pixels_per_cell : (rows, cols) of each cell.
    cells_per_block : (rows, cols) normalisation block size.
    target_size     : face crop size used during preprocessing.

    Returns
    -------
    np.ndarray, float32 – normalised HOG feature vector.
    """
    face = preprocess_image(source, target_size=target_size, grayscale=True)
    if face is None:
        raise ValueError("No face detected in image.")

    features = hog(
        face,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)

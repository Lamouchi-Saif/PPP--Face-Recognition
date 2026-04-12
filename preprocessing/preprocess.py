"""
Image preprocessing utilities:
  - load image from file or numpy array
  - convert to grayscale
  - normalise pixel values
  - resize to a fixed shape
  - detect and crop the largest face
  - denoise / histogram-equalise
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMAGE_SIZE, FACE_MARGIN, HAAR_CASCADE_PATH


def load_image(source) -> np.ndarray:
    """
    Load an image from a file path or return a copy if already a numpy array.

    Parameters
    ----------
    source : str | np.ndarray
        File path or BGR image array.

    Returns
    -------
    np.ndarray  BGR image (uint8).
    """
    if isinstance(source, np.ndarray):
        return source.copy()
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] (float32).
    If the image has integer dtype the values are divided by 255.
    """
    img = img.astype(np.float32)
    return img / 255.0


def resize(img: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize *img* to (width, height) using bilinear interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Apply a mild Gaussian blur to reduce sensor noise."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalisation to improve lighting invariance."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if gray.dtype != np.uint8:
        gray = (gray * 255).clip(0, 255).astype(np.uint8)
    return clahe.apply(gray)


def _get_haar_classifier() -> cv2.CascadeClassifier:
    """Return a face Haar Cascade classifier."""
    path = HAAR_CASCADE_PATH or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {path}")
    return clf


def detect_faces(img: np.ndarray):
    """
    Detect faces in *img* using the Haar Cascade detector.

    Parameters
    ----------
    img : np.ndarray
        BGR or grayscale image.

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of (x, y, w, h) bounding boxes, largest face first.
    """
    gray = to_grayscale(img) if len(img.shape) == 3 else img
    if gray.dtype != np.uint8:
        gray = (gray * 255).clip(0, 255).astype(np.uint8)

    clf = _get_haar_classifier()
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    # Sort by area descending (largest face first)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [tuple(f) for f in faces]


def crop_face(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    margin: float = FACE_MARGIN,
) -> np.ndarray:
    """
    Crop the face region from *img* with an optional margin.

    Parameters
    ----------
    img    : BGR or grayscale image.
    box    : (x, y, w, h) bounding box.
    margin : fractional padding added around the box (0 = no padding).
    """
    x, y, w, h = box
    h_img, w_img = img.shape[:2]

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    return img[y1:y2, x1:x2]


def preprocess_image(
    source,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    grayscale: bool = True,
    detect_face: bool = True,
    apply_equalize: bool = True,
) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline:
      load → (detect & crop face) → grayscale → denoise
      → histogram equalise → resize → normalise.

    Parameters
    ----------
    source       : file path or numpy array (BGR).
    target_size  : output (width, height).
    grayscale    : if True the output has a single channel.
    detect_face  : if True, attempt to crop the largest face.
    apply_equalize: if True apply CLAHE equalisation.

    Returns
    -------
    np.ndarray (float32, values in [0, 1]) or None if no face was detected
    when *detect_face* is True.
    """
    img = load_image(source)

    if detect_face:
        faces = detect_faces(img)
        if not faces:
            return None
        img = crop_face(img, faces[0])

    if grayscale:
        img = to_grayscale(img)

    img = denoise(img)

    if apply_equalize and grayscale:
        img = equalize_histogram(img)

    img = resize(img, target_size)
    img = normalize(img)
    return img

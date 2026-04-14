"""
Viola-Jones / Haar Cascade feature extraction.

The classifier returns a flattened histogram of the gray-level values inside
each detected sub-window (a simple but effective representation of the face
region after Viola-Jones detection).  When used as input to an SVM we
supplement it with the raw normalized pixel values of the resized face crop.
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMAGE_SIZE
from preprocessing.preprocess import (
    load_image,
    to_grayscale,
    detect_faces,
    crop_face,
    resize,
    normalize,
    equalize_histogram,
    denoise,
)


def extract_haar_features(source, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    Extract Haar-Cascade-based features from an image.

    Strategy
    --------
    1. Detect the largest face using the Haar Cascade detector.
    2. Crop and preprocess the face region.
    3. Return the flattened, normalised pixel vector as the feature vector.
       (This is the representation used when classifying with an SVM trained
        on Haar-detected face crops.)

    Parameters
    ----------
    source      : file path or BGR numpy array.
    target_size : (width, height) of the output crop.

    Returns
    -------
    np.ndarray, shape (width*height,) – float32 normalised values.

    Raises
    ------
    ValueError if no face is detected in the image.
    """
    img = load_image(source)
    gray = to_grayscale(img)

    faces = detect_faces(gray)
    if not faces:
        raise ValueError("No face detected in image.")

    face_crop = crop_face(gray, faces[0])
    face_crop = denoise(face_crop)
    face_crop = equalize_histogram(face_crop)
    face_crop = resize(face_crop, target_size)
    face_crop = normalize(face_crop)

    return face_crop.flatten().astype(np.float32)


def draw_faces(img: np.ndarray) -> np.ndarray:
    """
    Return a copy of *img* with bounding boxes drawn around all detected faces.
    Useful for visualisation / debugging.
    """
    out = img.copy()
    faces = detect_faces(out)
    for x, y, w, h in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

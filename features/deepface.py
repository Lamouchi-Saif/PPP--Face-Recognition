"""
deepface_utils.py
-----------------
Fonctions DeepFace utilisees par le programme principal.

Fonctions exportees :
    - load_deepface()                          -> bool
    - extract_embedding(bgr_img)               -> np.ndarray(128,) | None
    - cosine_similarity(a, b)                  -> float
    - euclidean_distance(a, b)                 -> float
    - verify_identity(bgr_img, known_embedding) -> dict
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Thresholds (can be overridden by importing from config if available)
# ---------------------------------------------------------------------------
try:
    from config import COSINE_THRESHOLD, EUCLIDEAN_THRESHOLD
except ImportError:
    COSINE_THRESHOLD    = 0.60   # cosine similarity > threshold  → same person
    EUCLIDEAN_THRESHOLD = 10.0   # euclidean distance < threshold → same person

# ---------------------------------------------------------------------------
# DeepFace import
# ---------------------------------------------------------------------------
_DeepFace = None

def load_deepface() -> bool:
    """
    Importe DeepFace et charge le modele Facenet en memoire.
    A appeler une seule fois au demarrage du programme principal.

    Returns:
        True  si DeepFace est disponible et charge avec succes.
        False sinon (bibliotheque non installee).
    """
    global _DeepFace
    if _DeepFace is not None:
        return True
    try:
        from deepface import DeepFace
        _DeepFace = DeepFace
        print("[deepface_utils] DeepFace charge avec succes.")
        return True
    except ImportError:
        print("[deepface_utils] DeepFace non installe. Executez : pip install deepface")
        return False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_embedding(bgr_img: np.ndarray,
                      model: str = 'Facenet',
                      detector: str = 'opencv') -> np.ndarray | None:
    """
    Extrait un embedding facial depuis une image BGR (format OpenCV).

    Le pipeline interne de DeepFace :
      1. Detection du visage (backend = detector)
      2. Alignement automatique via les landmarks
      3. Redimensionnement 160x160 + normalisation
      4. Passage dans le CNN Facenet → vecteur de 128 float32

    Args:
        bgr_img  : image numpy BGR uint8, telle que retournee par cv2.imread()
                   ou cv2.VideoCapture.read().
        model    : modele DeepFace a utiliser ('Facenet', 'ArcFace', etc.)
        detector : backend de detection ('opencv', 'retinaface', etc.)

    Returns:
        np.ndarray de shape (128,) et dtype float32 si un visage est detecte.
        None si aucun visage n'est trouve ou si DeepFace n'est pas disponible.
    """
    if _DeepFace is None:
        return None
    try:
        result = _DeepFace.represent(
            img_path=bgr_img,
            model_name=model,
            detector_backend=detector,
            enforce_detection=True,
        )
        # DeepFace >= 0.0.75 retourne une liste de dicts
        raw = result[0]['embedding'] if isinstance(result, list) else result['embedding']
        return np.array(raw, dtype=np.float32)
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcule la similarite cosinus entre deux embeddings.

    Formule : sim = (a . b) / (||a|| * ||b||)

    Args:
        a, b : vecteurs numpy de meme dimension (typiquement shape (128,)).

    Returns:
        float dans [-1.0, 1.0].
        1.0  = vecteurs identiques  → meme personne
        0.0  = vecteurs orthogonaux → personnes differentes
        -1.0 = vecteurs opposes
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcule la distance euclidienne (L2) entre deux embeddings.

    Formule : d = ||a - b||

    Args:
        a, b : vecteurs numpy de meme dimension (typiquement shape (128,)).

    Returns:
        float >= 0.
        Proche de 0  → meme personne
        Grand (> 10) → personnes differentes
    """
    return float(np.linalg.norm(a - b))


def verify_identity(bgr_img: np.ndarray,
                    known_embedding: np.ndarray,
                    model: str = 'Facenet',
                    detector: str = 'opencv') -> dict:
    """
    Compare le visage presente dans bgr_img avec un embedding de reference.
    C'est la fonction principale a appeler dans le programme de controle d'acces.

    Args:
        bgr_img         : image BGR uint8 de la camera ou d'un fichier.
        known_embedding : embedding de reference de la personne autorisee,
                          obtenu precedemment via extract_embedding().
        model           : modele DeepFace a utiliser.
        detector        : backend de detection.

    Returns:
        dict avec les cles :
            'match'      (bool)  : True si les deux visages appartiennent a la meme personne.
            'cosine'     (float) : similarite cosinus entre les deux embeddings.
            'distance'   (float) : distance L2 entre les deux embeddings.
            'embedding'  (np.ndarray | None) : embedding extrait de bgr_img, ou None si echec.
            'error'      (str | None) : message d'erreur si l'extraction a echoue.

    Example:
        ref_emb = extract_embedding(cv2.imread('alice.jpg'))
        frame   = cap.read()[1]
        result  = verify_identity(frame, ref_emb)
        if result['match']:
            print("Acces autorise")
        else:
            print(f"Acces refuse — similarite : {result['cosine']:.3f}")
    """
    query_emb = extract_embedding(bgr_img, model=model, detector=detector)

    if query_emb is None:
        return {
            'match':     False,
            'cosine':    0.0,
            'distance':  float('inf'),
            'embedding': None,
            'error':     'Aucun visage detecte dans l\'image fournie.',
        }

    sim  = cosine_similarity(query_emb, known_embedding)
    dist = euclidean_distance(query_emb, known_embedding)

    match = (sim >= COSINE_THRESHOLD) and (dist <= EUCLIDEAN_THRESHOLD)

    return {
        'match':     match,
        'cosine':    sim,
        'distance':  dist,
        'embedding': query_emb,
        'error':     None,
    }

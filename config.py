"""
Global configuration for the face recognition system.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "face_recognition.db")

# Haar Cascade XML shipped with OpenCV
HAAR_CASCADE_PATH = None  # resolved at runtime via cv2.data.haarcascades

# ── Image preprocessing ────────────────────────────────────────────────────────
IMAGE_SIZE = (128, 128)          # (width, height) used for resizing
FACE_MARGIN = 0.2                # fractional margin added around detected face

# ── Feature extraction ─────────────────────────────────────────────────────────
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

DEEPFACE_MODEL = "Facenet"       # DeepFace back-end: Facenet, VGG-Face, etc.
DEEPFACE_DETECTOR = "opencv"     # face detector used by DeepFace
EMBEDDING_DIM = 128              # Facenet output dimension

# ── SVM classifier ─────────────────────────────────────────────────────────────
SVM_KERNEL = "linear"  # "linear", "rbf", etc.
SVM_C = 1.0
SVM_GAMMA = "scale"

# ── Authentication thresholds ─────────────────────────────────────────────────
# Cosine similarity: 1 = identical; threshold above which access is GRANTED
COSINE_THRESHOLD = 0.60
# Euclidean distance: threshold below which access is GRANTED
EUCLIDEAN_THRESHOLD = 10.0

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY = "change-me-in-production"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

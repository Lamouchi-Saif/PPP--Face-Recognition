"""
Flask web application for face recognition authentication.

Routes
------
GET  /                     – Home page
GET  /register             – Registration form
POST /api/register         – Register a new identity (JSON or form data)
GET  /authenticate         – Authentication page (webcam)
POST /api/authenticate     – Authenticate a face (JSON or form data)
GET  /api/identities       – List registered identities
DELETE /api/identity/<name>– Remove an identity
GET  /results              – Show latest evaluation results (plots)
"""
import logging
import os
import sys
import base64
import json
import io
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from database.db import init_db, save_embedding, load_embeddings, list_identities, delete_identity,log_auth_attempt
from preprocessing.preprocess import load_image, preprocess_image
from features.haar_cascade import extract_haar_features
from features.lbp import extract_lbp_features
from features.hog_features import extract_hog_features
from authentication.authenticator import authenticate_svm, authenticate_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)

init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a base64-encoded data URL into a BGR numpy array."""
    import cv2

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 data.")
    return img


def image_from_request():
    """
    Extract an image (as BGR numpy array) from the current Flask request.

    Supports:
    - JSON body: {"image": "<base64 data URL>"}
    - Multipart form: file field named 'image'
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image")

        if not image_data:
            raise ValueError("Missing image field.")

        return decode_base64_image(image_data)
    
    if "image" in request.files:
        f = request.files["image"]
        if f and allowed_file(f.filename):
            buf = np.frombuffer(f.read(), np.uint8)
            import cv2
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode uploaded image.")

            return img

    raise ValueError("No valid image provided in request.")


def _deepface_available() -> bool:
    try:
        import deepface  # noqa: F401
        return True
    except ImportError:
        return False


def save_uploaded_image(img: np.ndarray, name: str) -> str:
    safe_name = secure_filename(name.lower().replace(" ", "_"))
    filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

    upload_dir = app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)

    abs_path = os.path.join(upload_dir, filename)
    cv2.imwrite(abs_path, img)

    return os.path.join("static", "uploads", filename).replace("\\", "/")

# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    identities = list_identities()
    return render_template("index.html", identities=identities,
                           deepface_available=_deepface_available())


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/authenticate")
def authenticate_page():
    return render_template("authenticate.html",
                           deepface_available=_deepface_available())


@app.route("/results")
def results_page():
    results_dir = os.path.join(config.BASE_DIR+"\static", "results")
    plots = []
    if os.path.isdir(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if f.endswith(".png"):
                plots.append(f)
    return render_template("results.html", plots=plots)


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/identities", methods=["GET"])
def api_list_identities():
    return jsonify(list_identities())


@app.route("/api/identity/<name>", methods=["DELETE"])
def api_delete_identity(name):
    delete_identity(name)
    return jsonify({"status": "ok", "deleted": name})


@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Register a new face identity.

    Body (JSON):
        { "name": "Alice", "image": "<base64 data URL>", "method": "deepface" }
    or multipart form with fields: name, method, image (file).
    """
    try:
        img = image_from_request()
    except Exception as exc:
        logger.debug("Image decode error in /api/register: %s", exc)
        return jsonify({"error": "Invalid or missing image."}), 400

    if request.is_json:
        data = request.get_json(silent=True) or {}
        name = data.get("name", "").strip()
        method = data.get("method", "deepface")
    else:
        name = request.form.get("name", "").strip()
        method = request.form.get("method", "deepface")

    if not name:
        return jsonify({"error": "Identity name is required."}), 400

    try:
        embedding = _extract_embedding(img, method)
        file_path = save_uploaded_image(img, name)      
        save_embedding(name, method, embedding, file_path)
        return jsonify({
            "status": "registered",
            "identity": name,
            "method": method,
            "image_path": file_path,
            "embedding_dim": len(embedding),
        })
    except Exception as exc:
        logger.error("Registration failed for %r: %s", name, exc)
        return jsonify({
            "error": "Registration failed. Please try again.",
            "details": str(exc),
        }), 500


@app.route("/api/authenticate", methods=["POST"])
def api_authenticate():
    """
    Authenticate a face.

    Body (JSON):
        { "image": "<base64 data URL>", "method": "deepface", "metric": "cosine" }
    """
    try:
        img = image_from_request()
    except Exception as exc:
        logger.debug("Image decode error in /api/authenticate: %s", exc)
        return jsonify({"error": "Invalid or missing image."}), 400

    if request.is_json:
        data = request.get_json(silent=True) or {}
        method = data.get("method", "deepface")
        metric = data.get("metric", "cosine")
    else:
        method = request.form.get("method", "deepface")
        metric = request.form.get("metric", "cosine")

    try:
        query_emb = _extract_embedding(img, method)
        db_names, db_embs = load_embeddings(method)
        if len(db_names) == 0:
            return jsonify({
                "authenticated": False,
                "error": f"No registered embeddings found for method '{method}'."
            }), 404
        result = authenticate_embedding(
            query_emb, db_names, db_embs, metric=metric
        )
        decision = "granted" if result.get("authenticated") or result.get("match") else "denied"

        log_auth_attempt(
            predicted_name=result.get("identity") or result.get("name"),
            method=method,
            decision=decision,
            score=result.get("score") or result.get("similarity"),
            distance=result.get("distance"),
            image_path=save_uploaded_image(img, "auth_attempt"),
        )
        return jsonify(result)
    except Exception as exc:
        logger.error("Authentication error: %s", exc)
        return jsonify({"error": "Authentication failed. Please try again."}), 500
    
    


def _extract_embedding(img: np.ndarray, method: str) -> np.ndarray:
    """Dispatch feature extraction based on *method*."""
    if method == "deepface":
        from features.deepface import load_deepface, extract_embedding
        
        if not load_deepface():
            raise RuntimeError("DeepFace is not installed or could not be loaded.")

        emb = extract_embedding(img)
        if emb is None:
            raise ValueError("No face detected by DeepFace.")
        return emb
    if method == "haar":
        return extract_haar_features(img)
    if method == "lbp":
        return extract_lbp_features(img)
    if method == "hog":
        return extract_hog_features(img)
    raise ValueError(f"Unknown method: {method!r}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=5000)

# Face Recognition & Authentication System

A complete face recognition pipeline that combines **traditional feature extraction** (Haar Cascade, LBP, HOG) with **deep learning** (DeepFace/Facenet) to authenticate users — exposed through a minimal **Flask web application**.

---

## Features

| Component          | Technology                                        |
| ------------------ | ------------------------------------------------- |
| Face detection     | Viola-Jones / OpenCV Haar Cascade                 |
| Feature extraction | LBP, HOG, Haar pixel vectors                      |
| Deep embeddings    | DeepFace (Facenet, 128-dim)                       |
| Classifier         | SVM (scikit-learn, RBF kernel)                    |
| Embedding store    | SQLite                                            |
| Evaluation         | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Web interface      | Flask + webcam capture                            |

---

## Project Structure

```
PPP--Face-Recognition/
├── app.py                  # Flask web application
├── train.py                # CLI training pipeline
├── config.py               # Global settings / thresholds
├── requirements.txt
│
├── preprocessing/
│   └── preprocess.py       # Load, detect, crop, normalise, resize
│
├── features/
│   ├── haar_cascade.py     # Viola-Jones feature extraction
│   ├── lbp.py              # Local Binary Patterns
│   ├── hog_features.py     # Histogram of Oriented Gradients
│   └── deepface_features.py# DeepFace embeddings + similarity helpers
│
├── classification/
│   └── svm_classifier.py   # SVM training, saving, inference
│
├── evaluation/
│   └── evaluate.py         # Metrics, confusion matrix, comparison plots
│
├── authentication/
│   └── authenticator.py    # SVM and embedding-based auth logic
│
├── database/
│   └── db.py               # SQLite CRUD for identities & embeddings
│
├── templates/              # Jinja2 HTML templates
│   ├── index.html
│   ├── register.html
│   ├── authenticate.html
│   └── results.html
│
├── static/
│   ├── css/style.css
│   └── js/webcam.js
│
└── tests/
    └── test_system.py      # Pytest test suite (25 tests)
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
# optional (heavy, ~1 GB models downloaded on first use):
pip install deepface
```

### 2 – Run the web application

```bash
python app.py
```

Open `http://localhost:5000` in a browser.

### 3 – Register a face

1. Navigate to **Register**.
2. Enter a name, choose a feature method, click **Start Camera**, then **Capture & Register**.
3. The embedding is stored in `face_recognition.db`.

### 4 – Authenticate

1. Navigate to **Authenticate**.
2. Click **Start Camera**, then **Authenticate** (or **Live** for continuous checks).
3. The system returns `ACCESS GRANTED` with the matched identity or `ACCESS DENIED`.

---

## Training pipeline (CLI)

Prepare a dataset directory:

```
data/
  Alice/
    img001.jpg
    img002.jpg
  Bob/
    img001.jpg
    ...
```

Then run:

```bash
python train.py --data_dir data --method all [--optimize]
```

This trains one SVM per feature method, evaluates on a held-out test split,
saves trained models to `models/`, and writes comparison plots to `results/`.

---

## Authentication logic

### SVM mode (Haar / LBP / HOG)

1. Extract features from the query image.
2. Pass through the trained SVM; read the top-class probability.
3. Grant access if probability ≥ configured confidence threshold.

### Embedding mode (DeepFace)

1. Extract a 128-dim Facenet embedding from the query image.
2. Compare with all stored embeddings using **cosine similarity** or **Euclidean distance**.
3. Grant access if the best score meets the configured threshold.

Thresholds are configurable in `config.py`:

```python
COSINE_THRESHOLD    = 0.60   # similarity >= threshold → GRANTED
EUCLIDEAN_THRESHOLD = 10.0   # distance  <= threshold → GRANTED
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

25 tests covering preprocessing, feature extraction (LBP, HOG), authentication
logic, database CRUD, evaluation helpers, and Flask endpoints — all run without
needing a real dataset or deep learning models.

---

## Technologies

- **Python** – OpenCV, NumPy, Pandas, Matplotlib
- **Machine Learning** – scikit-learn (SVM), scikit-image (LBP, HOG)
- **Deep Learning** – DeepFace (Facenet embedding & recognition)
- **Database** – SQLite (via Python's `sqlite3`)
- **Deployment** – Flask web framework

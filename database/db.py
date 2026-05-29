"""
SQLite database layer for storing face embeddings and identities.
"""
import sqlite3
import numpy as np
from config import DB_PATH

""" Database schema overview -----------------------------------

identities
   ↓--------has many
images
   ↓--------has many
embeddings (BLOB vectors)

"""


def get_connection():
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all required tables and indexes if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""

    -- ─────────────────────────────────────────────────────────────
    -- Identities (persons)
    -- ─────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS identities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- ─────────────────────────────────────────────────────────────
    -- Images (source data)
    -- ─────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        identity_id INTEGER NOT NULL,
        file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (identity_id)
            REFERENCES identities(id)
            ON DELETE CASCADE,
            UNIQUE(identity_id, file_path)
    );

    -- ─────────────────────────────────────────────────────────────
    -- Embeddings (feature vectors)
    -- ─────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        method TEXT NOT NULL ,
        embedding BLOB NOT NULL,
        dim INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id)
            REFERENCES images(id)
            ON DELETE CASCADE,
            UNIQUE(image_id, method)
    );
    -- ─────────────────────────────────────────────────────────────
    -- Authentication attempts (optional, for logging and analysis)
    -- ─────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS auth_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER,
    predicted_name TEXT,
    method TEXT NOT NULL,
    score REAL,
    distance REAL,
    decision TEXT NOT NULL,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_id)
        REFERENCES identities(id)
        ON DELETE SET NULL
    );

    -- ─────────────────────────────────────────────────────────────
    -- Indexes (performance optimization)
    -- ─────────────────────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_images_identity
        ON images(identity_id);

    CREATE INDEX IF NOT EXISTS idx_embeddings_image
        ON embeddings(image_id);

    CREATE INDEX IF NOT EXISTS idx_embeddings_method
        ON embeddings(method);
    
    CREATE INDEX IF NOT EXISTS idx_auth_attempts_identity
    ON auth_attempts(identity_id);

    CREATE INDEX IF NOT EXISTS idx_auth_attempts_method
        ON auth_attempts(method);

    CREATE INDEX IF NOT EXISTS idx_auth_attempts_created_at
        ON auth_attempts(created_at);
    """)

    conn.commit()
    conn.close()

# ── Identity helpers ───────────────────────────────────────────────────────────

def add_identity(name: str) -> int:
    """Insert a new identity or return existing id."""
    name = name.strip()

    if not name:
        raise ValueError("Identity name cannot be empty.")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO identities (name) VALUES (?)",
        (name,)
    )

    conn.commit()

    cursor.execute(
        "SELECT id FROM identities WHERE name = ?",
        (name,)
    )

    row = cursor.fetchone()
    conn.close()

    return row["id"]


def list_identities():
    """Return a list of all registered identities."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM identities ORDER BY name")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def delete_identity(name: str):
    """Remove an identity and all its embeddings (cascades)."""
    conn = get_connection()
    conn.execute("DELETE FROM identities WHERE name = ?", (name,))
    conn.commit()
    conn.close()


# ── Embedding helpers ──────────────────────────────────────────────────────────

def save_embedding(name: str, method: str, embedding: np.ndarray, file_path: str):
    """
    Save embedding with full pipeline:
    identity → image → embedding
    """
    identity_id = add_identity(name)
    image_id = add_image(identity_id, file_path)

    # Convert to float32 (optimized)
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding_bytes = embedding.tobytes()

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO embeddings (image_id, method, embedding, dim)
        VALUES (?, ?, ?, ?)
        """,
        (image_id, method, embedding_bytes, embedding.shape[0])
    )
    conn.commit()
    conn.close()


def load_embeddings(method: str):
    """
    Returns:
        names: list[str]
        embeddings: np.ndarray (N, D)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT i.name, e.embedding, e.dim
        FROM embeddings e
        JOIN images img ON e.image_id = img.id
        JOIN identities i ON img.identity_id = i.id
        WHERE e.method = ?
        """,
        (method,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], np.empty((0,0), dtype=np.float32)

    names = []
    vectors = []

    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        names.append(r["name"])
        vectors.append(vec)

    return names, np.vstack(vectors)

# This function is similar to load_embeddings but includes more metadata for each embedding.
def load_embeddings_with_metadata(method: str):
    """
    Return all embeddings for a method with identity and image metadata.
    Useful for authentication and result display.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            e.id AS embedding_id,
            i.id AS identity_id,
            i.name AS name,
            img.file_path AS file_path,
            e.method AS method,
            e.embedding AS embedding,
            e.dim AS dim,
            e.created_at AS created_at
        FROM embeddings e
        JOIN images img ON e.image_id = img.id
        JOIN identities i ON img.identity_id = i.id
        WHERE e.method = ?
        ORDER BY i.name, e.created_at
        """,
        (method,)
    )

    rows = cursor.fetchall()
    conn.close()

    results = []

    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)

        if vec.shape[0] != r["dim"]:
            raise ValueError(
                f"Corrupted embedding {r['embedding_id']}: "
                f"expected dim={r['dim']}, got {vec.shape[0]}"
            )

        results.append({
            "embedding_id": r["embedding_id"],
            "identity_id": r["identity_id"],
            "name": r["name"],
            "file_path": r["file_path"],
            "method": r["method"],
            "embedding": vec,
            "dim": r["dim"],
            "created_at": r["created_at"],
        })

    return results


def clear_embeddings(method: str):
    """Delete all embeddings for a given method."""
    conn = get_connection()
    conn.execute("DELETE FROM embeddings WHERE method = ?", (method,))
    conn.commit()
    conn.close()

# ── Images Helpers ───────────────────────────────────────────────────────────

def add_image(identity_id: int, file_path: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO images (identity_id, file_path)
        VALUES (?, ?)
        """,
        (identity_id, file_path)
    )

    conn.commit()

    cursor.execute(
        """
        SELECT id FROM images
        WHERE identity_id = ? AND file_path = ?
        """,
        (identity_id, file_path)
    )

    row = cursor.fetchone()
    conn.close()

    return row["id"]

# ── Advanced queries (optional) ─────────────────────────────────────────────────
def get_identity_embeddings(name: str, method: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT e.embedding
        FROM embeddings e
        JOIN images img ON e.image_id = img.id
        JOIN identities i ON img.identity_id = i.id
        WHERE i.name = ? AND e.method = ?
        """,
        (name, method)
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        np.frombuffer(r["embedding"], dtype=np.float32)
        for r in rows
    ]

def count_embeddings():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as c FROM embeddings")
    count = cursor.fetchone()["c"]

    conn.close()
    return count

# Additional helper to get identity details by name (for display or validation)
def get_identity_by_name(name: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, created_at FROM identities WHERE name = ?",
        (name.strip(),)
    )

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None

# Additional helper to count total identities (for stats or validation)
def count_identities():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS c FROM identities")
    count = cursor.fetchone()["c"]

    conn.close()
    return count

# Additional helper to count embeddings by method (for stats or validation)
def count_embeddings_by_method():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT method, COUNT(*) AS count
        FROM embeddings
        GROUP BY method
        ORDER BY method
        """
    )

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows

# Additional helper to log authentication attempts (for analysis and debugging)
def log_auth_attempt(
    predicted_name: str | None,
    method: str,
    decision: str,
    score: float | None = None,
    distance: float | None = None,
    image_path: str | None = None,
):
    conn = get_connection()
    cursor = conn.cursor()

    identity_id = None
    if predicted_name:
        cursor.execute(
            "SELECT id FROM identities WHERE name = ?",
            (predicted_name,)
        )
        row = cursor.fetchone()
        identity_id = row["id"] if row else None

    cursor.execute(
        """
        INSERT INTO auth_attempts
        (identity_id, predicted_name, method, score, distance, decision, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identity_id,
            predicted_name,
            method,
            score,
            distance,
            decision,
            image_path,
        )
    )

    conn.commit()
    conn.close()
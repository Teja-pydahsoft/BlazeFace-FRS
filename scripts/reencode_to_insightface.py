"""
Re-encode database face entries using InsightFace buffalo_l (512-d embeddings).
Creates a DB backup first. For each row in face_encodings with observed_dim != 512,
attempts to load the associated image_path and compute a new embedding.

Usage:
    python scripts/reencode_to_insightface.py

Notes:
- This script is conservative: it does not delete existing encodings. It appends new encodings
  with encoding_type='insightface'. Admins can later remove old encodings if desired.
- Requires InsightFace models and onnxruntime available in the environment.
"""
import os
import sqlite3
import io
import numpy as np
from app.core.database import DatabaseManager
from app.core.insightface_embedder import InsightFaceEmbedder
import shutil
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'blazeface_frs.db')
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'backups')


def backup_db():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(BACKUP_DIR, f'blazeface_frs.db.bak.reencode.{ts}')
    shutil.copy2(DB_PATH, dest)
    print('DB backup created at', dest)
    return dest


def get_face_rows(conn):
    cur = conn.cursor()
    cur.execute('SELECT id, student_id, encoding, encoding_type, image_path FROM face_encodings')
    return cur.fetchall()


def observed_dim_from_bytes(enc_bytes):
    try:
        arr = np.load(io.BytesIO(enc_bytes), allow_pickle=False)
        return int(arr.shape[0])
    except Exception:
        try:
            arr = np.frombuffer(enc_bytes, dtype=np.float64).astype(np.float32)
            return int(arr.shape[0])
        except Exception:
            return None


def main():
    if not os.path.exists(DB_PATH):
        print('DB not found at', DB_PATH)
        return

    backup_db()

    db = DatabaseManager(DB_PATH)
    # Initialize InsightFace embedder
    try:
        embedder = InsightFaceEmbedder(model_name='buffalo_l')
    except Exception as e:
        print('Failed to initialize InsightFace embedder:', e)
        return

    conn = sqlite3.connect(DB_PATH)
    rows = get_face_rows(conn)

    reencoded = []
    failed = []

    for rid, sid, enc_bytes, enc_type, img_path in rows:
        dim = observed_dim_from_bytes(enc_bytes)
        if dim == 512:
            continue  # already good
        if not img_path or not os.path.exists(img_path):
            failed.append((rid, sid, 'missing_image'))
            continue
        try:
            # Load image and detect face; prefer to run embedder on cropped face if possible
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                failed.append((rid, sid, 'imread_failed'))
                continue
            # InsightFace expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = embedder.detect_and_encode_faces(img_rgb)
            if not results:
                # try whole-image embedding
                emb = embedder.get_embedding(img_rgb)
                if emb is None:
                    failed.append((rid, sid, 'no_embedding'))
                    continue
                embedding = np.asarray(emb, dtype=np.float32)
            else:
                # pick the first embedding
                embedding = np.asarray(results[0][1], dtype=np.float32)

            # Add to DB using DatabaseManager API
            ok = db.add_face_encoding(sid, embedding, encoding_type='insightface', image_path=img_path)
            if ok:
                reencoded.append((rid, sid))
            else:
                failed.append((rid, sid, 'db_insert_failed'))
        except Exception as e:
            failed.append((rid, sid, str(e)))

    conn.close()

    print('Re-encoding complete. Reencoded:', len(reencoded), 'Failed:', len(failed))
    if reencoded:
        print('Reencoded rows (original id, student):')
        for r in reencoded:
            print(' ', r)
    if failed:
        print('Failed entries:')
        for f in failed:
            print(' ', f)

if __name__ == '__main__':
    main()

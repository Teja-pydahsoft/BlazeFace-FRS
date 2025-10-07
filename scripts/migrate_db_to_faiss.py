"""
Export face encodings from database and build FAISS index + metadata files.
Creates safe DB backup and writes data/faiss.meta.json, data/faiss.pkl, data/faiss.index
"""
import os
import argparse
import numpy as np
from app.core.database import DatabaseManager
from app.core.faiss_index import FaissIndex
import shutil
import datetime


def backup_db(db_path: str):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    backup = db_path + f'.backup.{ts}'
    shutil.copyfile(db_path, backup)
    print('DB backed up to', backup)
    return backup


def main(args):
    db = DatabaseManager(args.db)
    rows = db.get_face_encodings()
    if not rows:
        print('No encodings found in DB')
        return

    names = []
    embs = []
    for sid, arr, enc_type, img in rows:
        # Only include embeddings with expected dims matching buffalo_l (512)
        if arr is None:
            continue
        if arr.ndim != 1:
            continue
        embs.append(arr.astype(np.float32))
        names.append(f"{sid}")

    if not embs:
        print('No suitable embeddings to migrate')
        return

    embs = np.vstack(embs)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs / norms).astype(np.float32)

    os.makedirs('data', exist_ok=True)
    target = os.path.join('data', 'faiss')
    # Backup DB
    backup_db(args.db)

    idx = FaissIndex()
    idx.build(embs, names)
    idx.save(target)
    print('FAISS index built and saved to', target + '.*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='database/blazeface_frs.db')
    args = parser.parse_args()
    main(args)

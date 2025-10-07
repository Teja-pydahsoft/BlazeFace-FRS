"""
Scan the database face_encodings table, report encoding lengths and encoding_type mismatches.
Optionally auto-fix encoding_type values when `--auto-fix` is passed (non-destructive update after backing up DB).

Usage:
    python scripts/check_and_fix_encodings.py [--auto-fix]

This script is safe: it makes a backup copy of the DB before any modifications.
"""
import sqlite3
import io
import numpy as np
import argparse
import shutil
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'blazeface_frs.db')
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'backups')

# mapping of observed dimension -> presumed encoding_type
DIM_TO_TYPE = {
    128: 'standard',
    512: 'insightface',
    256: 'simple'
}


def scan_db():
    if not os.path.exists(DB_PATH):
        print('Database not found at', DB_PATH)
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, student_id, encoding, encoding_type, image_path, created_at FROM face_encodings')
    rows = cur.fetchall()
    results = []
    for row in rows:
        rec_id, student_id, enc_bytes, enc_type, img, created = row
        observed_dim = None
        decoded = False
        try:
            arr = np.load(io.BytesIO(enc_bytes), allow_pickle=False)
            observed_dim = arr.shape[0]
            decoded = True
        except Exception:
            try:
                arr = np.frombuffer(enc_bytes, dtype=np.float64).astype(np.float32)
                observed_dim = arr.shape[0]
            except Exception:
                observed_dim = None
        results.append({
            'id': rec_id,
            'student_id': student_id,
            'declared_type': enc_type,
            'observed_dim': observed_dim,
            'image_path': img,
            'created_at': created,
            'decoded': decoded
        })
    conn.close()
    return results


def backup_db():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(BACKUP_DIR, f'blazeface_frs.db.bak.{ts}')
    shutil.copy2(DB_PATH, dest)
    print('Database backup created at', dest)
    return dest


def auto_fix(results):
    # backup first
    backup_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fixes = []
    for r in results:
        rid = r['id']
        obs = r['observed_dim']
        declared = r['declared_type']
        if obs is None:
            continue
        guessed = DIM_TO_TYPE.get(obs)
        if guessed and guessed != declared:
            print(f"Fixing id={rid}: declared={declared} -> guessed={guessed} (dim={obs})")
            cur.execute('UPDATE face_encodings SET encoding_type = ? WHERE id = ?', (guessed, rid))
            fixes.append((rid, declared, guessed))
    conn.commit()
    conn.close()
    print(f"Applied {len(fixes)} fixes")
    return fixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-fix', action='store_true', help='Automatically fix declared encoding_type based on observed dimension (makes DB backup)')
    args = parser.parse_args()

    results = scan_db()
    if not results:
        print('No encodings found in DB')
        return

    print('Scan results:')
    for r in results:
        print(f"  id={r['id']}, student={r['student_id']}, declared={r['declared_type']}, observed_dim={r['observed_dim']}, decoded={r['decoded']}, img={r['image_path']}")

    mismatches = [r for r in results if r['observed_dim'] and DIM_TO_TYPE.get(r['observed_dim']) and DIM_TO_TYPE.get(r['observed_dim']) != (r['declared_type'] or '').lower()]
    if mismatches:
        print('\nMismatched records:')
        for r in mismatches:
            print(f"  id={r['id']}, student={r['student_id']}, declared={r['declared_type']}, observed_dim={r['observed_dim']}")
    else:
        print('\nNo obvious declared-type mismatches found')

    if args.auto_fix:
        fixes = auto_fix(results)
        if fixes:
            print('Fixes applied:')
            for f in fixes:
                print(' ', f)
        else:
            print('No fixes necessary')

if __name__ == '__main__':
    main()

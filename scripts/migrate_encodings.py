"""
Migration helper: convert existing face_encodings BLOBs to numpy .npy bytes format
- Makes a backup copy of the database file
- Reads all rows from face_encodings
- Attempts to decode existing BLOB; if it's raw float64 bytes, converts to float32 and saves as numpy bytes
- Updates the row with the new bytes

Usage:
    python scripts/migrate_encodings.py --db database/blazeface_frs.db

This script should be run in a staging environment first and a backup verified.
"""

import sqlite3
import argparse
import shutil
import os
import io
import numpy as np


def backup_db(db_path: str) -> str:
    backup_path = db_path + '.bak'
    shutil.copy2(db_path, backup_path)
    print(f"Backup created at: {backup_path}")
    return backup_path


def migrate(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    print(f"Starting migration on: {db_path}")
    backup_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, student_id, encoding FROM face_encodings")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Found {total} encodings to inspect")

    updated = 0
    skipped = 0

    for row in rows:
        row_id, student_id, encoding_blob = row
        try:
            # Try to load as numpy first (if already migrated this will succeed)
            buf = io.BytesIO(encoding_blob)
            arr = np.load(buf, allow_pickle=False)
            # If load succeeded and dtype is float32, assume migrated
            if arr.dtype == np.float32:
                skipped += 1
                continue
            else:
                arr = arr.astype(np.float32)
        except Exception:
            # Not numpy; try interpreting as raw float64 bytes
            try:
                arr = np.frombuffer(encoding_blob, dtype=np.float64)
                arr = arr.astype(np.float32)
            except Exception as e:
                print(f"Skipping id {row_id} student {student_id}: cannot decode ({e})")
                skipped += 1
                continue

        # Save arr to numpy bytes
        buf2 = io.BytesIO()
        np.save(buf2, arr, allow_pickle=False)
        new_bytes = buf2.getvalue()

        # Update DB
        cursor.execute("UPDATE face_encodings SET encoding = ? WHERE id = ?", (new_bytes, row_id))
        updated += 1

    conn.commit()
    conn.close()

    print(f"Migration completed. Updated: {updated}, Skipped: {skipped}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=False, default='database/blazeface_frs.db', help='Path to sqlite DB')
    args = parser.parse_args()
    migrate(args.db)

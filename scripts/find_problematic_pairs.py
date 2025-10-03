"""
Scan face encodings in the database to find potentially problematic high-similarity pairs
across different student IDs. Outputs CSV and writes findings to a new DB table
'problematic_pairs'.

This script is intended to be run in maintenance windows. The comparison is O(N^2)
so use --max-rows to limit scan size on large databases.

Usage:
    python scripts/find_problematic_pairs.py --db database/blazeface_frs.db --out problematic_pairs.csv --threshold 0.85 --max-rows 1000

"""

import argparse
import csv
import io
import os
import sys
import sqlite3
import numpy as np
from datetime import datetime


def ensure_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS problematic_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id_a TEXT NOT NULL,
            student_id_b TEXT NOT NULL,
            similarity REAL NOT NULL,
            encoding_type TEXT,
            encoding_id_a INTEGER,
            encoding_id_b INTEGER,
            image_path_a TEXT,
            image_path_b TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()


def load_encodings(conn: sqlite3.Connection, max_rows: int = None):
    cursor = conn.cursor()
    query = "SELECT id, student_id, encoding, encoding_type, image_path FROM face_encodings"
    if max_rows:
        query += f" LIMIT {int(max_rows)}"
    cursor.execute(query)
    rows = cursor.fetchall()

    encs = []
    for row in rows:
        enc_id, student_id, enc_bytes, enc_type, image_path = row
        # Try to load numpy binary format
        try:
            buf = io.BytesIO(enc_bytes)
            arr = np.load(buf, allow_pickle=False)
        except Exception:
            try:
                arr = np.frombuffer(enc_bytes, dtype=np.float64)
                arr = arr.astype(np.float32)
            except Exception:
                print(f"Skipping encoding id {enc_id} (failed to decode)")
                continue
        if arr is None or arr.size == 0:
            continue
        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(arr)
        if norm == 0:
            continue
        arr = arr.astype(np.float32) / norm
        encs.append((enc_id, student_id, arr, enc_type, image_path))
    return encs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main():
    parser = argparse.ArgumentParser(description="Find problematic high-similarity encoding pairs")
    parser.add_argument("--db", default="database/blazeface_frs.db", help="Path to SQLite DB")
    parser.add_argument("--out", default="problematic_pairs.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (cosine) to report")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit how many encodings to load")
    parser.add_argument("--min-diff-ids", action="store_true", help="Require student_id A != B to report pairs (recommended)")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    ensure_table(conn)

    encs = load_encodings(conn, max_rows=args.max_rows)
    total = len(encs)
    print(f"Loaded {total} encodings")

    results = []

    # O(N^2) scan, compare pairs i<j
    for i in range(total):
        id_a, sid_a, arr_a, type_a, img_a = encs[i]
        for j in range(i+1, total):
            id_b, sid_b, arr_b, type_b, img_b = encs[j]
            if args.min_diff_ids and sid_a == sid_b:
                continue
            if arr_a.shape != arr_b.shape:
                # skip different-sized encodings
                continue
            sim = cosine_similarity(arr_a, arr_b)
            if sim >= args.threshold:
                results.append((sid_a, sid_b, float(sim), type_a, id_a, id_b, img_a, img_b))

    # Write CSV
    with open(args.out, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["student_id_a", "student_id_b", "similarity", "encoding_type", "encoding_id_a", "encoding_id_b", "image_path_a", "image_path_b"])
        for row in results:
            writer.writerow(row)

    print(f"Found {len(results)} problematic pairs; wrote CSV: {args.out}")

    # Insert into DB
    cur = conn.cursor()
    for (sid_a, sid_b, sim, enc_type, id_a, id_b, img_a, img_b) in results:
        cur.execute("""
            INSERT INTO problematic_pairs (student_id_a, student_id_b, similarity, encoding_type, encoding_id_a, encoding_id_b, image_path_a, image_path_b)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid_a, sid_b, sim, enc_type, id_a, id_b, img_a, img_b))
    conn.commit()
    print("Inserted problematic pairs into DB")


if __name__ == '__main__':
    main()
